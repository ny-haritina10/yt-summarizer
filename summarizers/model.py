import logging
import requests
import time
import json
import re
import os
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
SUMMARIZATION_API_URL = f"https://api-inference.huggingface.co/models/{SUMMARIZATION_MODEL}"

REFINEMENT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
REFINEMENT_API_URL = f"https://api-inference.huggingface.co/models/{REFINEMENT_MODEL}"

summary_cache: Dict[str, str] = {}

CONFIG_FILE = 'config.json'
API_TOKEN = None 

def load_config() -> Optional[str]:
    """Loads the HF API token from config.json or environment variable."""
    global API_TOKEN
    
    if API_TOKEN:
        return API_TOKEN
    API_TOKEN = None

    try:
        script_dir = os.path.dirname(__file__)
        config_path = os.path.normpath(os.path.join(script_dir, '..', CONFIG_FILE))

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                loaded_token = config.get("hf_api_token")
                if loaded_token:
                    API_TOKEN = loaded_token
                    logger.info("Loaded HF API token from config.json.")
                    return API_TOKEN
                else:
                    logger.warning("HF API token not found or not set in config.json.")
        else:
            logger.warning(f"Configuration file not found at: {config_path}")

    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from configuration file: {config_path}")
    except Exception as e:
        logger.error(f"An error occurred loading configuration from file: {e}")

    logger.info("Attempting to load HF API token from environment variable HF_API_TOKEN.")
    env_token = os.environ.get("HF_API_TOKEN")
    if env_token:
        API_TOKEN = env_token
        logger.info("Loaded HF API token from environment variable HF_API_TOKEN.")
        return API_TOKEN
    else:
        logger.error("HF API token is missing. Please add it to config.json or set HF_API_TOKEN environment variable.")
        return None

def preprocess_text(text: str) -> str:
    """Basic text preprocessing: normalize whitespace."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text_by_chars(text: str, max_chunk_length_chars: int) -> List[str]:
    """Chunks text by characters, trying to respect sentence boundaries."""
    chunks = []
    current_start = 0
    while current_start < len(text):
        current_end = min(current_start + max_chunk_length_chars, len(text))
        chunk = text[current_start:current_end]
        if current_end < len(text):
            sentence_endings = ['.', '!', '?']
            last_sentence_end_index = -1
            for ending in sentence_endings:
                index = chunk.rfind(ending)
                if index > last_sentence_end_index:
                    last_sentence_end_index = index
            if last_sentence_end_index > len(chunk) * 0.7:
                current_end = current_start + last_sentence_end_index + 1
                chunk = text[current_start:current_end]
        chunks.append(chunk)
        current_start = current_end
    logger.info(f"Chunked text into {len(chunks)} chunks based on char limit {max_chunk_length_chars}.")
    return chunks

def query_hf_api(
    api_url: str,
    payload: Dict,
    api_token: str,
    retries: int = 3,
    delay: int = 15 
) -> Optional[Any]: 
    """
    Sends a request to a specified Hugging Face Inference API endpoint.

    Args:
        api_url: The specific API endpoint URL for the model.
        payload: The data to send in the request body.
        api_token: The Hugging Face API token.
        retries: Number of times to retry on specific errors (e.g., 503).
        delay: Seconds to wait between retries.

    Returns:
        The parsed JSON response from the API, or None on failure.
    """
    headers = {"Authorization": f"Bearer {api_token}"}
    logger.debug(f"Querying API: {api_url}")
    logger.debug(f"Payload keys: {payload.keys()}") 

    for attempt in range(retries + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=90) 
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON response from {api_url}. Response text: {response.text}")
                    return None 
            elif response.status_code == 503:
                logger.warning(f"API {api_url} returned 503. Retrying in {delay}s... (Attempt {attempt + 1}/{retries + 1})")
                if attempt < retries: time.sleep(delay)
                else: logger.error(f"Max retries reached for 503 error at {api_url}."); return None
            elif response.status_code == 429:
                logger.error(f"API Rate Limit Exceeded (429) for {api_url}. Cannot proceed.")
                return None
            elif response.status_code == 401:
                 logger.error(f"API returned 401 (Unauthorized) for {api_url}. Check your API token.")
                 return None
            elif response.status_code == 400: 
                 logger.error(f"API returned 400 (Bad Request) for {api_url}. Check payload/input length. Response: {response.text}")
                 return None
            else:
                logger.error(f"API request to {api_url} failed with status {response.status_code}: {response.text}")
                return None

        except requests.exceptions.Timeout:
             logger.error(f"API request to {api_url} timed out after 90 seconds.")
             if attempt < retries: logger.warning(f"Retrying in {delay}s..."); time.sleep(delay)
             else: logger.error("Max retries reached for timeout."); return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during API request to {api_url}: {e}")
            if attempt < retries: logger.warning(f"Retrying in {delay}s..."); time.sleep(delay)
            else: logger.error("Max retries reached for network error."); return None
        except Exception as e:
             logger.error(f"An unexpected error occurred during API query to {api_url}: {e}")
             return None 
    return None

def refine_summary_with_llm(
    summary_text: str,
    api_token: str,
    max_new_tokens: int = 250 
    ) -> Optional[str]:
    """
    Uses a text-generation LLM via API to refine a given summary.

    Args:
        summary_text: The initial summary text to refine.
        api_token: The Hugging Face API token.
        max_new_tokens: Max tokens the refinement model should generate.

    Returns:
        The refined summary text, or None if refinement fails.
    """
    logger.info("Attempting to refine summary using LLM...")

    prompt = (
        f"Please refine the following summary for clarity, organization, and readability. "
        f"Ensure it flows well and captures the key points accurately. "
        f"Format the output clearly, using paragraphs or bullet points if helpful.\n\n"
        f"Original Summary:\n{summary_text}\n\n"
        f"Refined Summary:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False, 
            "temperature": 0.7, # creativity
            "return_full_text": False 
        }
    }

    api_response = query_hf_api(REFINEMENT_API_URL, payload, api_token)

    if api_response:
        if isinstance(api_response, list) and len(api_response) > 0 and 'generated_text' in api_response[0]:
            refined_text = api_response[0]['generated_text'].strip()

            logger.info("Successfully refined summary.")
            return refined_text
        else:
            logger.error(f"Unexpected API response format during refinement: {api_response}")
            return None
    else:
        logger.error("Failed to get response from refinement API.")
        return None


def summarize_transcript(
    transcript: str,
    video_id: str, 
    max_summary_length: int = 150,
    min_summary_length: int = 30,
    max_chunk_chars: int = 4000,
    refine: bool = True 
) -> Optional[str]:
    """
    Summarizes transcript using API, optionally refines using another LLM API.

    Args:
        transcript: The transcript text to summarize.
        video_id: The unique ID of the video (for caching).
        max_summary_length: Target max length for initial summary (tokens, approx).
        min_summary_length: Target min length for initial summary (tokens, approx).
        max_chunk_chars: Max characters per chunk sent to the summarization API.
        refine: Whether to perform the LLM refinement step (default: True).

    Returns:
        The final (potentially refined) summary text, or None if summarization fails.
    """
    # 1. Load API Token
    api_token = load_config()
    if not api_token:
        logger.error("Cannot summarize: Missing Hugging Face API Token.")
        return None

    # 2. Check cache for the final (refined) summary
    cache_key = f"{video_id}_refined" if refine else video_id
    if cache_key in summary_cache:
        logger.info(f"Returning cached summary for key: {cache_key}")
        return summary_cache[cache_key]

    logger.info(f"Starting API-based summarization for video ID: {video_id} (Refine={refine})")

    # 3. Preprocess
    processed_text = preprocess_text(transcript)
    if not processed_text:
        logger.warning("Transcript is empty after preprocessing.")
        return ""

    # 4. Chunk text
    text_chunks = chunk_text_by_chars(processed_text, max_chunk_chars)

    # 5. Summarize chunks via Summarization API
    chunk_summaries = []
    total_chunks = len(text_chunks)
    for i, chunk in enumerate(text_chunks):
        logger.info(f"Summarizing chunk {i+1}/{total_chunks} via API...")
        if not chunk.strip(): continue

        payload = {
            "inputs": chunk,
            "parameters": {
                "max_length": max_summary_length,
                "min_length": min_summary_length,
                "do_sample": False
            }
        }
        api_response = query_hf_api(SUMMARIZATION_API_URL, payload, api_token)

        if api_response:
            if isinstance(api_response, list) and len(api_response) > 0 and 'summary_text' in api_response[0]:
                chunk_summaries.append(api_response[0]['summary_text'])
                logger.info(f"Successfully summarized chunk {i+1}/{total_chunks}.")
            else:
                logger.error(f"Unexpected API response format for chunk {i+1}: {api_response}")
                return None 
        else:
            logger.error(f"Failed to get summary for chunk {i+1}/{total_chunks}. Aborting summarization.")
            return None

    # 6. Combine chunk summaries
    initial_summary = " ".join(chunk_summaries).strip()
    if not initial_summary:
         logger.warning("Initial summary is empty after combining chunks.")
         return "" 

    logger.info(f"Successfully generated initial summary via API for video ID: {video_id}")

    # 7. Refine summary using LLM API (if requested)
    final_summary = initial_summary
    if refine:
        refined_summary = refine_summary_with_llm(initial_summary, api_token)
        if refined_summary:
            final_summary = refined_summary
        else:
            logger.warning("Refinement step failed. Using the initial summary.")
            cache_key = video_id 
            
    # 8. Update cache with the final result
    logger.info(f"Caching final summary under key: {cache_key}")
    summary_cache[cache_key] = final_summary
    return final_summary

if __name__ == '__main__':
    loaded_token = load_config()
    if not loaded_token:
         print("CRITICAL: API Token could not be loaded. Please check config.json or HF_API_TOKEN env var. Exiting test.")
         exit()
    
    masked_token = loaded_token[:4] + "****" + loaded_token[-4:]
    print(f"Using API Token: {masked_token}")

    example_transcript = """
        Cal aai is really cool like I downloaded that app and it's amazing but there's going to be like a hundred of them in 
        the App Store in like two months and then Cal will not be able to charge what they are and I just go like those are 
        good oneoff bites those are good for going from a college dorm to being a millionaire but they're not going to get 
        you to a decamillionaire I think like your Equity value is just toast within a very short period of time because I just 
        think the pricing ultimately goes to zero what do you see as like a business that you can Implement AI or build an


        agent that can still be highly competitive and successful in two or three years if you want to build an AI 
        business you need either a data advantage or a network effect Advantage it's never been easier to build a 
        million-dollar business and it's never been easier to lose all your Equity value so I just think that's a hard 
        world to be a business person and it's exciting it's the most excited I've ever been for business because it's easier 
        than ever to start a business but the worst businesses are the ones that are easy to


        start good to see you good to see you too man so today I was thinking we can

        Business building in AI

        talk about uh where are you seeing the opportunities right now with AI because in my mind I you know I know you look up 
        to the Charlie mongers and War buffets of the world value investing you know Warren Buffett and Charlie 
        Munger value investing you know what does that mean it means like seeing a rough in the diamonds and and sort of 
        seeing opportunity and I'm curious where are you seeing opportunity in this world of AI agents and and AI in general I


        think the hard part is that um in a lot of instances it's like a sustaining Innovation right and what I mean by that 
        is it's a new technology that will drop the cost so let's say that you're um you know we were just talking about 
        recruiting businesses so let's say um you know you start a online recruiting business or you buy an online recruiting 
        business and they have a 20 person outbound email team so you know you might think you're a genius because 
        you're like oh I'll get a you know an outbound recruiter agent that will just replace all those people and now my


        margins will be way higher but what actually happens is maybe for 6 months you get that Arbitrage but all of your 
        competitors have the same realization within 6 12 months months and then the price drops and you're in the same 
        business and maybe it's even more competitive than before so I think um it's one of the most interesting times 
        to be an investor and entrepreneur because you know we've always been building sand castles on the beach in 
        Tech like I think we all knew um you know anyone can compete you know you start a to-do list software business and


        anyone at any time can launch a competitor it's just that before there was a finite number of coders in the 
        world uh there was only so much Venture Capital money out there and there was just less competition I think now you 
        get far more competition in Far More niches um and I think Moes matter more than ever you know where you know a mo 
        before might have been um you know I make funeral home management software and the moat is that it's too small of a


        market for Venture Capital to play in and none of the other people that own funeral homes know how to code but 
        that's not not a mode anymore what a real mode would be is I have software that allows um funeral homes to keep 
        track of all their inventory and trade it back and forth or buy it and then you have a network effect now someone can't 
        just go Vibe code that in a weekend because now you have a network effect so um you know I think social networks are 
        really interesting I think communities are really interesting but I think most tools have become a 100 times harder


        yeah I mean I I totally agree I think uh I just want to I want to get more tactical I'm trying to think like what 
        okay I think everyone agrees honestly I think everyone agrees that I mean I can't tell you how many distribution is 
        the new moat tweet I've seen in the last like three weeks it's like everyone has finally agreed that distribution is the 
        new Mo that anyone could create anything the latest thing is by the way that Vibe coding is really don't Vibe code your 
        way to production code that's like the latest meme that's going around cuz I don't know if you saw there was this


        like viral post that this guy was using cursor and he ended up posting it uh production code and then he got hacked 
        and bad things could happen um so I agree with all those things and I think 99% of people 
        listening to this agree so what's our next step like I actually want to treat this as 
        like you know and we've done this before I've just you know you've called me I've called you and been like okay so what do 
        we do here so I've got I mean I've got a bunch of ideas I wrote down a bunch of ideas for things people could build as


        as we usually do the problem you know as I look at it is almost all of them can be competed Away really really quickly 
        so I think it's been it's never been easier to build a million-dollar business and it's never been easier to 
        lose all your Equity value right so I think it's going to be really easy to have gold rush uh you know oh hey like 
        Peter levels can launch this flight imator and make a million dollar but is that a sustainable business in two years 
        when there's a thousand of these flight simulators to be honest I don't know what the opportunity is


        um I think it's a bit of a sad answer like I I'm not seeing a lot of opportunities to go and buy or start 
        businesses um where I think there's a sustainable competitive Advantage I think there's a cost Advantage there's a 
        Time Advantage but I'm not I mean I would be really curious to put that back to you like what do you see as like a 
        business that you can Implement AI or build an agent that can still be highly competitive and successful in two or 
        three years because I'm increasingly of the mind that for all tools um basically all middlemen


        businesses are gone and I believe fundamentally that we are all going to go to the search bar and the new search 
        bar is chat gbt and you're going to query it and chat gbt will just build whatever you need or it'll hook into 
        whatever apis you need and so the only things that really matter at this point now are um you know real world goods and 
        services uh compute or network effects of some kind I I don't like that answer what what have you got like I'd love to 
        hear your answer to this I mean my take is if you want to build an AI business you need either a data advantage or a


        network effect Advantage so um I think what does that mean for us it means that we probably shouldn't be 
        buying or incubating AI businesses today so quick break in the Pod to tell you a little bit about startup Empire so 
        startup Empire is my private membership where it's a bunch of people like me like you who want to build out their


        startup ideas now they're looking for content to help accelerate that they're looking for potential co-founders 
        they're looking for uh tutorials from people like me to come in and tell them how do you do email marketing how do you 
        build an audience how do you go viral on Twitter all these different things that's exactly what startup Empire is 
        and it's for people who want to start a startup but are looking for ideas or it's for people who have a startup but 
        just they're not seeing the traction uh that they need so you can check out the link to Startup empire.co in the


        description I think the advant I think

        Media Business Opportunities and AI

        what I what I'm interested in doing frankly and and I'm giving my playbook here but it's it's basically like I 
        would I would either want to buy media businesses or incubate media businesses why 
        because I think that if you assume that you can Vibe code software and you can create the most beautiful to-do list app 
        I saw someone uh Vibe code their way to things three a clone of things 3 which is something that I've used for for many 
        years now in minutes and it like actually works and it's beautiful so if you can assume that okay you can Vibe


        code things three which for me was like an aha moment around like beautiful software to-do list app then okay the 
        opportunities in either buying media businesses incubating media businesses and then once you've built that Network 
        effect then you can start saying like Okay now how do I get these people's data that's your next step your first 
        step is how do I get your the attention next step is how do I get your data and then the third step is okay how can I 
        build assistance co-pilots AI software to actually monetize and build a sustainable


        business so what would be the sort of media business that you would want to buy because I maybe I have a different 
        take but I want to hear what you're imagining like what let's say you could press a button and buy magically buy the 
        company of your choice what company would it be well I'll tell you like my framework for thinking about what would 
        be a good company to buy so I want it to be a high value Niche like I wouldn't want it to be you know I don't en I I 
        envy Mr Beast in a lot of ways but on on on the other hand I don't envy Mr Beast like he's got billions of people looking


        at him but they're they're low value I'm just talking purely economical like I want a he's got to sell stuff to kids 
        and kids don't have money exactly um but I'll I'll I'll give one I I'll give two two businesses that I think 
        would be interesting to buy that are high value that are probably under monetized one is Tech runch which just 
        has a great brand name to you and I like we have so many memories from you know the Silicon Valley early Silicon Valley


        days like launching on Tech runch that sort of thing where I think it was bought by 
        AOL uh right yeah I think so Michael Arrington was the guy who started it and then he sold in like what 2008 or 
        something exactly so I bet you there's a deal to be had where they're probably struggling they're trying to probably 
        figure out who they are what they are um but they've got a high value value Niche with a great brand so I would buy 
        something like Tech runch the other kind of the other interesting thing to buy would be a very sought-after event


        series so my dream you ask who my dream company would to buy would be like if I can buy 
        South by Southwest you know Ted is for sale right now maybe you should buy that I didn't 
        know is is that is that for real yeah it's for it's for real I think it's public knowledge right now they're 
        shopping it around and he's gonna give it I think he's gonna give it away to whoever is going to be the best new 
        owner of it so maybe you should go pitch Chris Chris Anderson yeah I mean I think uh I think


        Ted has done an incredible job of turning an event series into a media company but they've stopped that Media 
        company so I think this is a perfect like Ted would be incredible around like okay all these people who watch Ted Ted 
        videos ideas worth spreading how do you give them not just ideas but also tools and what are those tools and how could 
        you VI Vibe code your so just just so I understand so let's say you buy um a yachting website and it has a big


        community of people that are all into yachting and they just read the news on there and you're basically saying you 
        could Vibe code a bunch of cool data tools so for example what are people paying for their yachts and how much are 
        they paying to maintain them and then once you have that data you can sell them services and all that kind of stuff 
        exactly yeah I think that's that's really smart it one I mean we've looked at a lot of um different media Brands 
        and you know buying these kind of distressed assets and stuff and we've always passed because it feels like um


        it feels like like do you you think about like what's happened to like Forbes now it's like do you remember 
        like seeing friends post on Forbes you be like whoa like that's crazy they're on Forbes and then you realize like 
        forbs basically just opened it up to anybody and any SEO person could just post on Forbes and it just kind of died 
        like I'm really curious like I love Kevin Rose but like relaunching dig I'm like so curious about that like does 
        that actually work and is that Nostalgia worth something um and I I I'm fully with you in terms of like taking a


        community that is monetized in the wrong way and just monetizing it in the right way um do you think that's a unique AI 
        related thing though because I think that was an opportunity five years ago and it's still will be now it'll just be 
        a little easier to build it'll be easier and cheaper to build the tools to sell them yeah I think uh it was an 
        opportunity back then two years ago three years ago but I think that it's a it's not a little bit easier to 
        build it's it's significantly easier to build and also maybe even more importantly there's demand from


        consumers to try new apps for example the one the the flavor right now is Cal you know apps like that like those those 
        companies are doing like $20 million ARR on on downloading an app and you know how hard it is to download apps and 
        getting someone to to to to download an app so I think that uh what AI has done also has just made it 
        um it's it's it's created new use cases it's created you know with C for example you take a picture and it tells you


        using llms how many calories it is that's cool you couldn't do that a few years ago you know it's interesting um I 
        got to have dinner with Charlie Munger like for the first time like six years ago and 
        you know he's like at the time he was like 97 and we would we would be talking about all these different ideas for 
        Investments and stuff and he would always just say like it's hard investing is hard and I was kind of like you know 
        it was like a 2018 or whatever like I was at my peak in terms of like you know no one else was doing what we're doing


        we're just like on a buying frenzy we're you know valuations are great and I was like what is this old guy talking about 
        like and what I realized in retrospect is is that he you know he just got such a big scale and such a competitive part 
        of the market where you know you're buying businesses for 10 billion dollars to move the needle that it just becomes 
        inherently competitive and I think that is happening in in all arenas right now and I think I kind of am becoming I'm 
        slowly becoming the Crusty old man at the table where you know think about like all the things that were kind of


        easy to start before like you know newsletters and all that kind of stuff I just think there will be almost 
        unlimited competition in all these Arenas um whether it's apps like right now like the example is like Cal AI is 
        really cool like I downloaded that app and it's amazing uh and there's tons of those but there's going to be like a 
        hundred of them in the App Store in like two months and then Cal will not be able to charge what they are and I just go 
        like those are good oneoff bites those are good for going from a college dorm to being a millionaire but they're not


        going to get you to a decamillionaire I think like your Equity value is just toast within a very short period of time 
        because I just think the pricing ultimately goes to zero um and I think all as well for like newsletters and 
        attention in general like I recently I own like a a local newsletter business and I was goofing around with Lindy and 
        I was able in about an hour to build a a Lindy agent that could go through um it would basically like choose a I I chose 
        a city um like a city near me and I said every day I want you to uh there's three different agents one sources all the


        news and it goes through like local Reddit like forums local news websites or whatever and then uh another one that 
        like takes it all and writes it and punches it up and makes it like fun and cool or whatever and then the other one 
        that formats it as a newsletter and it is perfect it is better than any local news letter that I could write and it's 
        more detailed because it goes to all these different sources and it even fact checks itself so I just think it's going 
        to become crazy I it's going to I think all of our inboxes are going to get overloaded with great content that's


        written by Ai and I think a lot of these businesses are just going to get so freaking hard like what the other thing 
        that's interesting is like when you talk about data like I find that whole thing really interesting because I think data 
        and a network effect are kind of interl where um you know I was talking to someone at um at one of the frontier 
        models I won't say who but I was like um oh you know we own all these social networks or whatever and they have all 
        this interesting proprietary data and he was like oh yeah we probably ate all that like you know they just they've


        gobbled up the entire internet and like yeah you can go and like sue them but like even a data you know it's not like 
        a data Moe because now all your data probably exists within an llm already right like I it's hard to identify like 
        what is actually unique data because all they need to do is gobble up a big enough sample size to have it be 
        predictable in almost any

        The future of GPT wrappers

        Arena I mean okay let's talk about I want to talk about a few things because you talked about a lot of things here so 
        the CI example if I was CI what I would have tried to so I would have acknowledged the fact that they're going 
        to get a thousand competitors and I would have said okay how do I create a network effect in this 
        app it doesn't look like they're doing that I I don't know but it doesn't look like they're doing that 
        so all the people that have downloaded Cal are people interested in health and tracking


        their health healthy lifestyle and maybe they want to lose weight so does the modern Weight Watchers which was a 
        multi-billion dollar company that like Oprah was a part of um does the modern Weight Weight Watchers start looking 
        like a cal ey in 2025 probably but they had to build the network effects to go there but Weight Watchers started as a 
        physical Moe because you'd go to a meet up in your town and then you'd have all this social buyin and you have people


        that are in your network selling you right I think Cal or these these guys just like I just don't know what my 
        advice would be like if that guy came to me I'd probably be like just sell like find someone who doesn't understand Ai 
        and just like sell it and be like look like the AR keeps going up you should buy this for $5 million and and ghetto 
        because uh personally like I tried Cal but I also think I tried like two or three others and I think I ultimately 
        found a tool that was just cheaper and like maybe I'm just a cheap bastard but like I think they basically get


        bifurcated into um you know there's all these like AI photo editors um on iPhone and they're just for suckers right like 
        it's literally like do a 7-Day free trial and then it's like $49 every two weeks and you're suddenly spending like 
        $1,500 a year for like a basic AI photo editor there's those guys and they're like borderline criminal like I find 
        that like crazy um and then there's people that'll just create these things for free or a dollar and and then 
        ultimately Apple will just eat them and this is this is the thing is like the largest the at the end of the day like


        the people that will ultimately capture a lot of this value are going to be whoever controls the devices because I 
        do believe like a lot of these llm um a lot of this functionality will Ely be implemented so I think like if you can 
        find a niche like let's talk about two different things so one is um buying a business or starting a business that is 
        a five or a 10 year business and then separately we can talk about how to make your first million bucks because I think


        like there's nothing wrong with making your first million bucks and building a Vibe coded crappy app and and then it 
        disappearing into years that's awesome that's a great win but for guys like you and me like it's got to be something 
        that is sustainable um yeah why don't we talk about should we should we dig into a couple ideas and stuff first of all

        Automation and AI tools we use

        before we dig into that I'm just curious we should talk about like what kind of automations and tools we're using uh I'd 
        love to hear what you're doing right now um so right now like currently I'm spending 
        a lot of time with gum Loop have you have you played with gum loop at all so gum Loop if understand it's similar to 
        like make or Lindy yeah exactly it it's very similar and by the way like I don't it's not like I'm recommending one


        platform or another platform like my my Mo right now is just to get smart on all the platforms and I'm starting to notice 
        that there's something I like more in Lindy and there's something I like more of gum Loop like there's something I 
        hate about gum Loop that Lindy or someone else does better so my recommendation honestly is just to try 
        them all and see what you like and you don't like um but I'm uh you know my company has been 
        running on zapier for the last four you know four years and zapier has been really good to us and and when I found


        zapier it was a GameChanger honestly for our business we were just automating so much 
        stuff that uh we had human beings do so for example um 
        sales leads like we would have someone go in every day look into our sales CRM and basically report back who are these 
        leads is this a good lead is this a bad lead and then create reports um with zapier which was really


        cool we created a slack Channel and every lead basically gets scored and gets put into the slack Channel and the 
        whole team gets to see uh the sales lead so it's to me I'm looking at that as like you know co-owner of the business 
        I'm like this is amazing but the the problem was the there really wasn't much intelligence 
        meaning uh it was more of like the scoring wasn't very good basically it's Boolean like if this then that if it


        doesn't match perfectly it's messed up it's messed up so what it basically did was it just posted it to a slack Channel 
        um with gum Loop now we can actually go and say like okay this sales lead came in Andrew wi Andrew 
        tiny.com um who is Tiny uh what time did he uh come in was this at two in the morning maybe people at two in the 
        morning actually want our services way more maybe we can charge them more because he came in at two in the morning


        um what is the size of tiny um how many companies do they own what is the revenue how many employees 
        um and it and just add intelligence in that in that process so from a sales perspective doing that with gum Loop has 
        been pretty fun and do you guys do um so let's say a lead comes in do you have an AI agent like email them back and say 
        like hey can you tell me more about your company and stuff exactly that's cool that's really cool yeah but only if it's


        not maybe a let's say we would score you as a 10 on 10 lead someone like you actually might not 
        want an agent to reach out to you you might actually a fake person and it's got to 
        be delayed a certain amount of time I know but I also think that uh you know people are saying that agents are going 
        to automate 20 everything 247 um there's some things that the human touch is going to be better right 
        like me recording a vid if you if you you know signed up to one of our forms maybe me actually recording a video and


        being like hey this is Greg you know we have a common friend in Shan pory um I'd love to talk to you about our service 
        that probably would be better at this stage with a human in the loop in my opinion totally but I do I do believe 
        like pretty soon Greg can be an agent and just be a video that looks like you yeah yeah I 
        agree so I've been messing around with that that's top of mine right now and then the other thing that's top of mine


        and I want to hear what you you've been up to is just Manis aai playing with Manis AI it's crazy it's so cool I just 
        I my um I I literally was I always have this problem where I get excited about a a project and then I go on Twitter and 
        I'm like hey I need someone to run this thing or I'm hiring for this role and then I get like 50 emails and so I did 
        that recently with a project which I'll talk about later and I got so many applicants and I just dread it because I 
        just I hate going through all the applicants and if I hand it off to my assistant like she kind of she's busy


        with other stuff and like she doesn't really know what to look for or whatever and so I took all those emails I 
        exported them out of Gmail as a zip and I uploaded 50 candidates to Manis and then it just spent 30 minutes scoring 
        them all based on what I needed and it was like these are the best three people and then I just focused on those people 
        and it was totally bang on so cool yeah I mean and it's unbiased too right even if it I'm not saying your 
        bias but everyone has some biases to any of their decisions subconscious um or not so I think that


        that's also really cool is I mean I guess you can say the llms have biases too though they have human I guess they 
        have human bias because they're trained on our data exactly so yeah I'll tell you kind of like a sampling of the 
        things I'm doing like one of the dumbest automations I built recently but I find so satisfying is I have a Lindy agent 
        that looks at my calendar events when I create them and it adds an emoji which is so simple but now my calendar looks 
        really beautiful um more impressive than that I built a Lindy agent that kind of does what you're talking about so it


        ingests every single email that I get and then it labels it or archives it um based on context and I have all these 
        examples of like different kinds of emails and whatever and then if it's something that I'm likely to say to 
        it'll actually draft the no email for me and I've tried like superum and Sara and a bunch of these other ones but I 
        ultimately wanted to roll my own and just have it just know exactly how it worked um and that's been amazing um the 
        other tool I've been using a lot is versel vzer if you messed with that dude crazy like I I own a um a pressure


        washing business locally and it's like you know I run it with some kid locally like it's just for for like kind of like 
        a hobby project or whatever um I just don't have time to like dig in and make the website better but they did this 
        like version him and a guy that worked for me did this version of the website which is like it's fine but it could 
        just be so much better and I just threw into into vzer and in like 30 seconds it had completely punched up the website 
        and we had code for it like to me that is just so mindboggling that is something I previously would have spent


        like a month honestly maybe doing um so that that's been really cool but the thing I really want to talk about and go 
        deep on in this episode is mCP um I think it's so cool um I've been uh implementing a bunch of that stuff Uh 
        custom and I actually think that might be like a my first million kind of opportunity for somebody um to really do 
        mCP right uh and I'm curious are you using mCP for anything right now I I haven't no I'm not I am not there yet so


        um yeah why don't we dive in I've got I've got a bunch of ideas and stuff but

        Startup Idea and Financial Analysis

        the first one is um so mCP stands for model what is it model context protocol and basically it's a it's a way of 
        piping in data or apis into an llm so the llm can basically query your data set and so um I had this like total 
        breakthrough with Claude maybe like three months ago um one of the things I love about a is it's like I find it's 
        like a bicycle for productivity like I just get so much more done and one of the main reasons I get more done is


        because I don't have to ask questions of other people so one of the things that would often happen to me would be um you 
        know it's it's a Sunday I'm working um and I I want to know um hey what's happening in this business with our 
        margin or how much are we spending on slack or you know just these random questions and those would be things that 
        unless I really wanted to get fiddly and dive into a bunch of accounting or whatever I would just email to somebody 
        I would lose interest in that problem let's say like I was cost cutting or something like that I'd lose interest in


        the problem because I have to wait four days for the answer and then I'd find out on Thursday um what I what I did is 
        I went to Claude And I said hey um what information would you need from zero to be able to give me like a full financial 
        analysis on my business and so it gave me a list and you know took like 20 minutes but I went and I exported all 
        the um you know CSV files out of zero I trained a cloud project and it was basically on my accounting for all my 
        personal holding company and my personal life for the last two years and it was magic like just absolute magic like at


        one point I said um How Could You optimize my taxes and it said oh you have a credit line over here and you 
        have a credit line in your company if you use the um company money to pay down the personal credit line you can just 
        immediately write off like it's like I think it was $100,000 of tax savings or something so that to me that was like oh 
        my God this is Magic um and when I learned about mCP I saw that as an opportunity to try and get live data 
        because the annoying thing about the cloud project is it's all historical data and it's all got to be manually


        updated and so I hired a developer to work with me literally like came to my house and went on my computer and set it 
        all up or whatever but we got it working and it's really cool so now I can say um it's linked to all my all the companies 
        that use certain like QuickBooks or zero or whatever and I can query and I can say something like um you know across 
        all my companies what's happening with revenue or how much are we spending on this thing or whatever query it is and 
        it can make graphs and charts and tables and stuff um and it's not perfect like I'd say like sometimes the numbers are


        wrong but it's a breakthrough and so the opportunity I think here is to make the easy kind of oneclick way to do this so 
        an mCP server for the Mac um where you can basically say I want to integrate all these apis and then it just manages 
        downloading the databases onto your computer it manages the interface with Claud and I know there's a few people 
        that have kind of tried to do this but I don't think anyone's really done it properly and and a service business


        around it like I might not want right like I might not want to hire developer to do this myself and sort of figure it 
        out but if someone could go and probably honestly charging monthly you probably can get away with charging monthly for 
        something like this well the best way to do it would be to be the competitive Advantage would be it's all about 
        security so you're the most secure you're the default like think about like one password is not hard like anyone can 
        Vibe code one password what matters about one password is everybody knows one password has not been compromised


        and they trust and that's why they pay you know I think we pay 200 bucks a month or whatever to use that software 
        so um I think that's a huge opportunity if someone can actually do it properly and it might be a good one-year business 
        it could also be a good five-year business if executed well I'll give away the name for this business so I would 
        branded ascp.com so think of it think about because mCP is a protocol just like HT 
        HTTP um but you know you also have HTT CPS I think the S stands for secure so if you could basically own I'm the


        secure mCP guy everywhere else is not secure and create that positioning dichotomy I think that that itself is 
        like going to help you win deals and I think if uh you want to own that domain you got to buy it off Greg for $20,000 
        because I assume he just bought it no I I do this on the p a lot where I give away domains and then within the first 
        like five minutes of posting someone Snipes it I don't know how you you do this you people but I don't even care


        like I just I want someone to go and do this build it and and pay it back somehow I like that I really like that 
        idea if anyone wants to do that we should uh we should back them not not that they need not that they need our 
        money yeah but you know it could be it could be fun it could be fun to to be involved so to your earlier point we 
        have distribution so we actually bringue to the table exactly we're not just we're not just 
        cranky old guys we we we we do bring distribution okay so that that's a

        Startup Idea Web Design Agency

        really good idea also I was thinking about what you were saying about your pressure washer business and I've got 
        another idea for someone to steal that I think is a simple how to make your first million 
        so I think that uh what what people should be doing is be using someone should create essentially an automated 
        agency built on top of vzer that scrapes local businesses 
        and prioritizes the ones that have ugly websites uh train it on like modern design practices


        minimalists uh you know make them beautiful build the websites and then automatically reaches out to these small 
        business owners and be like hey I just redid your website I love that click here to buy click here to buy and I'll 
        host it and then it's just 20 bucks a month to host it and then any changes you want are $10 every change exactly 
        that and we have your credit card on file they're happy because they've got this be they think they're getting a 
        steal right they're like oh my God $500 for this website and it's already done and they're GNA host it for me this is


        incredible that the bar is so low like it's crazy the services businesses I mean there's people making a fortune and 
        they have these terrible websites and they barely do SEO and stuff and I think there's a big opportunity just to take 
        best practices and roll them out in that Arena I love that though as a first business I think that's an incredible 
        idea and it might actually be a good long-term business at least for a while yeah have you looked into web hosting 
        businesses ever they're I mean they're phenomenal they're just like people it's like an accounting uh accounting firm


        it's like people just don't want to switch web hosts that often it's like look at GoDaddy like GoDaddy is the 
        perfect Testament to why it's a good business you can have that horrible user experience in that fugly website and 
        still be like a billion dollar company totally side note I was telling my wife I was like how do you you know how you 
        know someone's in older than 30 is if they want to register a domain and they go to GoDaddy 100% And they or like you 
        email them and you go to their website and it's like this this domain is parked by GoDaddy yeah exactly so okay I got

        Startup Idea Maxing

        another idea here um this one is like kind of Half Baked but I think it's kind of interesting so um I was dealing with 
        a difficult person via text message and I was like man like I really want help with this like I want help and and 
        sometimes I'll like take an email thread and I'll paste it into Claud and get feedback or whatever but I I started 
        thinking I was like I really wish that I had a way to analyze all my text messages and somebody somebody posted


        publicly and said you know with these 2 milli context Windows you can now um digest five years worth of text threads 
        and so my idea so like basically like um you know this whole like looks maxing Trend so like basically like it's these 
        apps where you can like take a photo of yourself and it'll like basically hotter not you it'll be like oh you know you 
        your skin is kind of bad and your hairlines receding and like here's all the things you can do and then you keep 
        taking photos to look better and so my idea is message maxing so so on on your Mac and this is just for Mac I don't


        know how this would work on PC but on your Mac iMessage syncs to a database called I think it's called chats. DB or 
        something like that and it's a big huge 2 gigabyte uh sqlite database my idea is you basically have something to ingest 
        that database and analyze it and it might be with a local llm because it might be too much of a pain in the ass 
        to upload it um and it basically um tells you it could tell you a bunch of different things but one example would 
        be it could tell you which of your friends or contacts is exhibiting toxic narcissistic or Psychopathic traits like


        who should you steer clear of who's being manipulative um another one might be like how are you exhibiting negative 
        behaviors yourself or how are you being kind of like beta or weak in dating and interactions or that sort of thing um 
        and then also like which friends are you neglecting who should you catch up with and I think like some of these things 
        Apple will do over time like I think their message summaries is like an indication that eventually they're going 
        to start digesting more of your messages but I don't think that Apple would ever tell you who's toxic or narcissistic or


        whatever and I just think this is the sort of thing that could go very very viral I think this is a one-year 
        business I think this is easy to copy I think someone will copy this idea when I share it they'll make a million dollars 
        it's not a long-term business but it's a cool idea yeah I so I started thinking about 
        businesses like this where I I I start with the Tik Tok because you can imagine what the Tik Tok 
        is going to be right like a viral Tik Tok am I a narcissist showing the screen oh my God you know my


        boyfriend's a narcissist like you know it's going to work it's just a matter of figuring out the right format so I think 
        that I call these businesses format businesses so it's it's a game of figuring out the format I agree with you 
        that you know it might only last a short amount of time but um it probably falls 
        into the c c well I think it's um I think it so I think and this kind of feeds into my 
        other idea so the other and I think what we said before of like if it's highly highly secure then it can be a high


        quality business right so if you think about other applications so those are kind of the fun genen Z ideas right of 
        like oh who's a narcist whatever but I think this would be useful for me in terms of business I think it could be 
        helpful of like who are you who are you messaging that you're turning off without realizing it or who's being 
        passive aggressive because I do a lot of business over text messages right having someone coach me on how to write more 
        effective text messages is just like how I'm using Ai and emails so like when I write emails I'm like a you know I'll


        write back an email that just says like sounds good and some people will view that as passive aggressive but if I add 
        a happy face then it's great or if I write hey Greg sounds good thanks Andrew right and so I just think something that 
        can like tweak what all of your interactions in a positive way I think could be really cool but yeah it's not 
        like a great longterm one it's kind of this idea is kind of like grammarly but more for like social 
        relationships well just think about what data is in there like I think messages are so much more intimate than even


        email and what I'm craving like I don't know that Google will do this um and this is another idea I have is like an 
        llm that actually ingests not only your messages but your email and then can actually tell you like what's going on 
        in your life so like when I go to chat GPT like you can say something there's a cool prompt that's like going around on 
        Tik Tok that's like based on what you know about me tell me what um tell me what psychological insights you can tell 
        me right and so I did it and it was like oh you're you're a Searcher you're always searching for the next thing or


        you know whatever it is and it's got all these like different insights and they're kind of like it's almost like 
        horoscopes but they're kind of real right I actually really I did this thing recently where I told Lindy to look at 
        my last 250 emails I had sent and then say uh what projects am I working on and what's the status of each and it was 
        pretty crazy how well it was able to summarize everything that I'm working on based on my email and so I think that um 
        I don't know who's going to do it but I think whoever does that will do very very well and I think it's probably


        Google but apparently the processing required to do it is quite difficult so the idea the the like startup idea there 
        would be to build um so what I did is I used Lindy and so Lindy sends it to an llm and reads every email some people 
        are very concerned about security they don't want to do that I think in the same way this like secure um 
        mCP idea someone could build a secure local email processor that ingests all your emails tells you what's going on


        and labels them and does all that on device yeah I think and and that's why I actually think this could be a bigger 
        business like you you can EV you can actually create a secure holding company you know you only work on secure 
        products and agencies and service businesses around you know AI meets SEC sec security um probably do pretty well

        Startup Idea Based Lending Solutions

        okay um okay my last idea so have you ever got a loan from A bank yeah so like it takes forever like whether 
        it's bank or private credit or any of these people it's like you know just to get a call is a week and then it's 
        another two months of like hey can you give us this document or that document and then all they do is they just like 
        package it up and they send it to like some wonk at head office who like reviews it all and that is basically an 
        llm right so my idea is AI based lending and the idea would be like you choose a vertical so you'd say like um I want to


        lend to you know SAS companies or whatever it is and it's just like hey how much do you want and then it starts 
        asking questions and it might say oh do you use stripe okay link us to your stripe API hey upload your ID upload 
        your Articles of Incorporation and it basically just automates all the diligence and packages it up and then at 
        the end you still have a human who goes and verifies that you know there's nothing sketchy or whatever it is but to 
        me this seems like a very obvious idea and it may already exist but I just haven't yet seen that like if I could go


        and get a $5 million Loan in 48 hours for like buying a business that would be incredible and I don't know why that's 
        not possible other than the fact that it's this very Antiquated industry okay I have a an idea similar to this I want 
        to share so did you see at shl sahill linia tweeted uh I want to give $100,000 to any Vibe coders 
        that will give me 10% profit in exchange for that did you see this no h when viral thought it was really interesting


        so here's the idea a little mix of sahil's idea and your idea it's the bank of V coding.com 
        and it's it's basically this it's an AI L lending llm where you take a percentage of of course you you know 
        you're not going to accept everyone right but you there's criteria in the llm that says like okay you know who who 
        are you know have you maybe launched successful projects in in the past have you uh do you have more than 10,000


        followers on x um is the project are the projects that you're going to work on have stripe 
        integration of course you need that um so and then you just take a percentage of profits I love that b right yeah 
        super smart I mean any of these these are the things that like previously someone would say this to me and I would 
        just say like no because it's an administrative Nightmare and you need to have an army of accountants and lawyers 
        to do it all and it just wouldn't make sense but I think now it totally makes sense to the point of like if you'd


        pitched me on the website hey we'll approach all these small businesses with predesigned websites I'd just be like oh 
        that's a nightmare business but with an agent that's a great business totally okay so you you started off this 
        conversation grumpy Charlie Munger and you've now ended the conversation no no no to be clear I'm 
        still saying almost all these businesses will be zeros in five years right right but but there's a possibility that 
        there's something and I think that you know I would say there there's a 20 or a 10% chance like five years ago pre AI of


        starting a business and having success and now I'm just like oh I think it's like a a three to five% for most 
        businesses because I think the competition is just so profound the counterargument is it's so 
        hard to predict even five months from now that how are we going to predict what's going to happen in five years 
        from now well I just think um you know the Dario amade quote of like you know uh all keyboard jobs are gone within two 
        years like if that is true then like I think like all all businesses are hard right I'm I'm more just going you know


        that maybe that's not a foregone conclusion I'm just saying hard is coming competition is coming and I think 
        the moment that someone can just prompt an llm to compete with you and like let's say I could go to an llm like 
        Manis and I could say hey Manis um I want you to build a calorie tracker to compete with Cal and I want you to beat 
        it and that's all you have to do which is totally possible in my opinion I don't see why that couldn't get to the 
        point of being real within two to three years so I just think that's a hard world to be a business person in it's


        exciting it's the most excited I've ever been for business because it's easier than ever to start a business but the 
        worst businesses are the ones that are easy to start that's why I think audience brand is really important like 
        for example Bank of vibe coding that's just sticky so you and I have personal monopolies right one of the reasons I've 
        spent so much time over the last couple years building an audience is because distribution like you said is valuable 
        and no one if someone resonates with you or me or Sam par or whoever then they're going to stick around and you know keep


        consuming your content so I think that's an unfair competitive Advantage yeah 
        exactly um also with that Vibe coding idea the Bank of vibe coding you're also if you build it now 
        you're integrated right and if you get a th000 Vibe coders to go build stuff and you own 10% of a thousand Vibe coders 
        businesses like that might be enough to set you up for Life yeah go V VCB it's like uh svb except uh Vibe coding bank I


        like it you should do that I would I would love to invest in that if you want a 
        partner um i' would love to partner with you on anything man awesome that'd be great um before we head out I'm just

        Andrew's Stealth Startup

        curious I know you are building an app I think it's called Vibe is that what it's called Yeah so basically um 
        the quick version on it is I used to dress like a total Schmo and then I got divorced and I was like okay I need to 
        dress better I hired a personal stylist which sounds really like fancy but it's actually not very expensive and it was 
        great she'd send me like nice outfits to wear and stuff but I'd have to literally text this woman and be like Oh should I


        wear this shirt with this or whatever and so I started just uploading photos to chat gbt and I'd be like hey I want 
        to wear these jeans today like what goes with it and then as I'd get dressed I'd just send more and more photos and it 
        would be like oh tussle your hair and like um wear this watch that you have and then I got more advanced I started 
        prompting it with all the clothes I had um and then I was like oh this is like a cool thin rapper and so I hired a agency 
        that I know and for you know 30 grand we threw together this app called Vibe it's in the App Store anyone can go look at


        it and I'm actually on the hunt that was the when I when we were talking earlier about uh Manis digging through all the 
        applicants so I had about 50 people apply I've got a couple candidates that are at the top but I need someone to run 
        this thing so I would love if any of uh any of you folks out there listening would want to run this uh I you have to 
        bootstrap it or raise money for it I'm not going to put in any money uh you need to not need a salary but I'll give 
        you tons of equity that's the gist and how do people apply um just email me cool I won't I


        won't say in my email they figure it out it's pretty easy cool all right I won't I won't include your email in the show 
        notes but uh comment comment below also if what you think of the idea and and uh Andrew thanks again for 
        go ahead oh yeah the I mean the cool the uni unique thing about it is you take photos of all your wardrobe and then 
        it'll tell you like hey these are outfits that'll work so you be like I'm going on a date night and it's a summer 
        night what should I wear and then it'll Bas it all off your wardrobe question on on this business


        idea is this a CI idea or is this a a big business in five 100% this is a cal ey this is like you're I I want like 
        some college kid who's like I want to get in and get out I want to come in I want to get it to a couple million 
        dollars of AR I want to cash flow it and then maybe it's a thing in two years but I have a feeling this is like a moment 
        in time I appreciate that Honesty by the way that's a depressing pitch I just want someone to do something with it but 
        I don't know if it's a big long-term opportunity yeah cool I appreciate the honesty Andrew this is you have an open


        invite to come back whenever you want whenever you want to be optimistic pessimistic God ideas Trends you know 
        where to find us thanks for coming on and sharing uh sharing some ideas yeah man thanks for having me talk to later 
        later
    """
    example_video_id = "jwst_api_example_refined"

    print("\n--- Testing API Summarization with Refinement ---")
    summary = summarize_transcript(
        example_transcript,
        example_video_id,
        max_summary_length=200, # Target for BART summary
        min_summary_length=50,
        max_chunk_chars=3000, # Chars per BART call
        refine=True
    )

    if summary:
        print("\nOriginal Length (chars):", len(example_transcript))
        print("Final Summary Length (chars):", len(summary))
        print("\nGenerated Summary (Refined):")
        print(summary)
    else:
        print("Summarization failed.")

    # Test caching (should hit cache now)
    print("\n--- Testing Caching (Refined) ---")
    summary_cached = summarize_transcript(example_transcript, example_video_id, refine=True)
    if summary_cached:
        print("Retrieved refined cached summary successfully.")
    else:
        print("Cache test failed.")

    print("\n--- Testing API Summarization WITHOUT Refinement ---")
    summary_no_refine = summarize_transcript(
        example_transcript,
        example_video_id, # Use same ID, but different cache key internally
        max_summary_length=200,
        min_summary_length=50,
        max_chunk_chars=3000,
        refine=False
    )
    if summary_no_refine:
         print("\nGenerated Summary (Initial - No Refinement):")
         print(summary_no_refine)
    else:
         print("Summarization without refinement failed.")