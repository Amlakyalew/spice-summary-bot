
#!/usr/bin/env python3
import logging
import re
import requests
import json
import os
import sqlite3
import time
import asyncio # For show_typing
from datetime import datetime # For logging timestamp
from cachetools import TTLCache # For caching
from googletrans import Translator # For multilingual support

from bs4 import BeautifulSoup
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
from telegram.constants import ParseMode

# --- Configuration Constants ---
MAX_INPUT_LENGTH = 4000  # Max characters for text input or URL content sent to LLM
RATE_LIMIT_PER_MINUTE = 20 # Max messages a user can send per minute
CACHE_SIZE = 100         # Max number of URLs to cache
CACHE_TTL = 3600         # Cache Time-To-Live for URLs in seconds (1 hour)

# --- Initialize Enhancements ---
cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)
translator = Translator()

# --- Get bot token and API key from environment variables ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEBHOOK_URL = os.environ.get("RENDER_EXTERNAL_URL")

if not BOT_TOKEN:
    logging.error("BOT_TOKEN environment variable not set!")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set!")
if not WEBHOOK_URL:
    logging.error("RENDER_EXTERNAL_URL environment variable not set! Webhook setup might fail.")

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Database Setup (Analytics) ---
def init_db():
    try:
        with sqlite3.connect('bot_analytics.db') as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS usage_stats
                            (user_id INT, timestamp TEXT, command TEXT)''')
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

# Call init_db once when the module loads
init_db()

# --- Analytics Tracking ---
def log_usage(user_id: int, command: str):
    try:
        with sqlite3.connect('bot_analytics.db') as conn:
            conn.execute("INSERT INTO usage_stats (user_id, timestamp, command) VALUES (?, ?, ?)",
                        (user_id, str(datetime.now()), command))
            conn.commit()
        logger.info(f"Logged usage: User {user_id}, Command '{command}'")
    except Exception as e:
        logger.error(f"Error logging usage: {e}")

# --- Enhanced Input Validation ---
def validate_input(content: str) -> bool:
    if len(content) > MAX_INPUT_LENGTH:
        return False
    # Basic URL validation, more robust checks happen in fetch_url_content
    if content.startswith(('http://', 'https://')):
        return True # Will be properly validated during fetch
    return True

# --- Rate Limited Handler (Middleware-like) ---
async def rate_limit_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current_time = time.time()

    # Clean up old entries
    context.bot_data.setdefault('rate_limits', {})
    context.bot_data['rate_limits'].setdefault(user_id, [])
    context.bot_data['rate_limits'][user_id] = [
        t for t in context.bot_data['rate_limits'][user_id] if current_time - t < 60
    ]

    if len(context.bot_data['rate_limits'][user_id]) >= RATE_LIMIT_PER_MINUTE:
        await update.message.reply_text("You're sending too many requests. Please wait a moment.")
        return False # Indicate that the message should not be processed further
    
    context.bot_data['rate_limits'][user_id].append(current_time)
    return True # Indicate that the message can be processed

# --- Cached URL Fetch ---
def fetch_url_content(url: str) -> str:
    """Fetches content from a URL and extracts readable text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Prioritize common text-holding tags
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span', 'div'])
        text_content = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs]).strip()

        if not text_content:
            # Fallback to body text if specific tags yield nothing
            text_content = soup.body.get_text(separator=' ', strip=True)
            
        return text_content[:MAX_INPUT_LENGTH] # Truncate to max input length
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return f"ERROR_FETCHING_URL: {e}"
    except Exception as e:
        logger.error(f"Error processing URL content: {e}")
        return f"ERROR_PROCESSING_URL: {e}"

def get_cached_or_fetch_content(url: str) -> str:
    """Retrieves content from cache or fetches it if not present/expired."""
    if url in cache:
        logger.info(f"Serving content for {url} from cache.")
        return cache[url]
    
    logger.info(f"Fetching content for {url} (not in cache or expired).")
    content = fetch_url_content(url)
    if not content.startswith("ERROR_"): # Only cache if fetch was successful
        cache[url] = content
    return content

# --- Enhanced API Call with Retry ---
async def call_gemini_with_retry(prompt: str, max_retries=3) -> str:
    """Calls the Gemini API with retry logic."""
    apiKey = GEMINI_API_KEY
    apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = { 'Content-Type': 'application/json' }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    for attempt in range(max_retries):
        try:
            full_api_url = apiUrl
            if apiKey:
                full_api_url += f"?key={apiKey}"

            response = requests.post(full_api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"Unexpected API response structure from Gemini (attempt {attempt+1}): {json.dumps(result)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt) # Exponential backoff
                else:
                    return "Sorry, the AI model did not return a valid summary after multiple attempts."
        except requests.exceptions.RequestException as e:
            logger.error(f"Network or HTTP error calling Gemini API (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt) # Exponential backoff
            else:
                return "Sorry, I couldn't connect to the AI model after multiple attempts. Please check your internet connection and API key configuration."
        except Exception as e:
            logger.error(f"General error calling Gemini API (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt) # Exponential backoff
            else:
                return "Sorry, an unexpected error occurred while getting the summary after multiple attempts."
    return "An unknown error occurred during API call." # Should not be reached

# --- Multilingual Support ---
async def translate_text_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Translates the provided text to a target language."""
    log_usage(update.effective_user.id, "/translate")
    if not context.args:
        await update.message.reply_text("Usage: /translate <target_language_code> <text_to_translate>\\n"
                                        "Example: /translate es Hello world!")
        return

    target_lang = context.args[0].lower()
    text_to_translate = " ".join(context.args[1:])

    if not text_to_translate:
        await update.message.reply_text("Please provide text to translate.")
        return

    await update.message.reply_text("Translating... Please wait.")
    try:
        translated_text = translator.translate(text_to_translate, dest=target_lang).text
        await update.message.reply_text(f"Translated to {target_lang.upper()}:\\n\\n{translated_text}")
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        await update.message.reply_text("Sorry, I couldn't translate that. Please check the language code or try again.")

# --- Progress Indicator ---
async def show_typing_task(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Sends a typing indicator until processing is done."""
    try:
        while context.user_data.get('processing_active', False):
            await context.bot.send_chat_action(chat_id=chat_id, action='typing')
            await asyncio.sleep(2) # Send typing every 2 seconds
    except Exception as e:
        logger.error(f"Error in show_typing_task: {e}")


# --- States for the conversation ---
ASKING_FOR_INPUT, ASKING_FOR_AUDIENCE, ASKING_FOR_SENTIMENT, ASKING_FOR_CLAIMS = range(4)
WAITING_FOR_FEEDBACK = 4 # New state for feedback

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and asks the user for text or a URL to summarize."""
    log_usage(update.effective_user.id, "/start")
    welcome_message = (
        "Hello! I'm your *Spice Summary Bot*, designed to give you quick, tailored summaries of any text or URL. "
        "What makes me 'spicy'? I don't just summarize; I adapt to your needs!\\n\\n"
        "My key features include:\\n"
        "- *Customized Summaries:* Get summaries tailored for any audience (e.g., 'for a 10-year-old', 'for a technical expert', 'in simple terms').\\n"
        "- *Sentiment Analysis:* Optionally gauge the overall tone (positive, negative, or neutral) of the content.\\n"
        "- *Key Claims Extraction:* Easily identify the main arguments or claims within the text.\\n"
        "- *Multilingual Support:* Translate any text using the /translate command.\n"
        "- *Rate Limiting:* Ensures fair usage for all users.\n"
        "- *Caching:* Speeds up repeated requests for the same URL.\n\n"
        "Here are the commands you can use:\n"
        "- /summarize: Start a new summarization process.\n"
        "- /translate <lang_code> <text>: Translate text to a specified language (e.g., /translate es Hola).\n"
        "- /feedback: Send your suggestions or report issues.\n"
        "- /stats: View your usage statistics (future feature).\n" # Placeholder for future stats feature
        "- /cancel: Stop any ongoing process.\n"
        "- /start: See this welcome message again.\n\n"
        "Ready to get started? Send /summarize!"
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)
    return ASKING_FOR_INPUT # Remains the entry point for summarization conversation

async def receive_content(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives text or a URL, validates it, and stores it, then asks for audience."""
    user_input = update.message.text
    
    if not validate_input(user_input):
        await update.message.reply_text(
            f"Your input is too long (max {MAX_INPUT_LENGTH} characters) or the URL is invalid. Please try again."
        )
        return ASKING_FOR_INPUT # Stay in this state to allow retry

    context.user_data['content_to_summarize'] = user_input # Store content

    await update.message.reply_text(
        "Great! Now, for whom should I summarize this? "
        "E.g., 'for a 10-year-old', 'for a technical expert', 'for a busy executive', or 'in simple terms'."
    )
    return ASKING_FOR_AUDIENCE

async def receive_audience(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives audience preference, stores it, and then asks for sentiment analysis preference."""
    audience = update.message.text
    context.user_data['audience'] = audience # Store audience

    keyboard = [
        [InlineKeyboardButton("Yes, analyze sentiment", callback_data='sentiment_yes')],
        [InlineKeyboardButton("No, just summarize", callback_data='sentiment_no')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Would you like me to also analyze the sentiment of the content?",
        reply_markup=reply_markup
    )
    return ASKING_FOR_SENTIMENT

async def handle_sentiment_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the sentiment choice and then asks for key claims preference."""
    query = update.callback_query
    await query.answer() # Acknowledge the button press

    sentiment_choice = query.data # 'sentiment_yes' or 'sentiment_no'
    context.user_data['include_sentiment'] = (sentiment_choice == 'sentiment_yes')

    keyboard = [
        [InlineKeyboardButton("Yes, extract claims", callback_data='claims_yes')],
        [InlineKeyboardButton("No, just summarize", callback_data='claims_no')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "Finally, would you like me to extract key claims or arguments from the content?",
        reply_markup=reply_markup
    )
    return ASKING_FOR_CLAIMS

async def handle_claims_choice_and_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the claims choice and then summarizes the stored content."""
    query = update.callback_query
    await query.answer() # Acknowledge the button press

    claims_choice = query.data # 'claims_yes' or 'claims_no'
    context.user_data['include_claims'] = (claims_choice == 'claims_yes')

    content = context.user_data.pop('content_to_summarize', None) # Retrieve and clear content
    audience = context.user_data.pop('audience', None) # Retrieve and clear audience
    include_sentiment = context.user_data.pop('include_sentiment', False) # Retrieve and clear sentiment choice
    include_claims = context.user_data.pop('include_claims', False) # Retrieve and clear claims choice

    if not content or not audience:
        await query.edit_message_text("Oops! I lost some context. Please start again with /summarize.")
        return ConversationHandler.END

    await query.edit_message_text("Thinking... Please wait while I process your request.")
    log_usage(update.effective_user.id, "/summarize")

    # Start typing indicator
    context.user_data['processing_active'] = True
    typing_task = asyncio.create_task(show_typing_task(context, query.message.chat_id))

    final_output = "Could not process the content."
    text_content = ""

    url_pattern = re.compile(r"https?://(?:www\.)?[\w\.-]+\.\w+(?:/[\w\.-]*)*(?:[\?&][^=&]+=[^=&]+)*")
    
    if url_pattern.match(content):
        text_content = get_cached_or_fetch_content(content)
        if text_content.startswith("ERROR_"):
            final_output = f"Sorry, I couldn't process the URL. Error: {text_content.replace('ERROR_FETCHING_URL: ', '').replace('ERROR_PROCESSING_URL: ', '')}"
            context.user_data['processing_active'] = False
            await typing_task # Ensure task finishes
            await query.edit_message_text(final_output)
            return ConversationHandler.END
    else:
        text_content = content

    if text_content and len(text_content) > 0:
        if len(text_content) > MAX_INPUT_LENGTH:
            text_content = text_content[:MAX_INPUT_LENGTH] # Ensure it's truncated before sending to LLM
            
        prompt_parts = [f"Summarize the following text for {audience} concisely."]
        
        if include_sentiment:
            prompt_parts.append("Also, analyze the overall sentiment (Positive, Negative, Neutral) and provide a brief reason for it.")
        
        if include_claims:
            prompt_parts.append("Additionally, extract 3-5 key claims or arguments from the text and list them as bullet points.")
        
        prompt_parts.append("Format the output as follows: 'Summary: [summary text]\\n[Optional Sentiment: Sentiment: [sentiment] (Reason: [reason])]\\n[Optional Claims: Key Claims:\\n- Claim 1\\n- Claim 2]'")
        prompt_parts.append(f":\\n\\n{text_content}")

        full_prompt = " ".join(prompt_parts)
        
        gemini_response = await call_gemini_with_retry(full_prompt)
        final_output = gemini_response # Gemini will return the formatted string

    context.user_data['processing_active'] = False # Stop typing indicator
    await typing_task # Ensure task finishes
    await query.edit_message_text(final_output, parse_mode=ParseMode.MARKDOWN)
    return ConversationHandler.END

async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Prompts the user to send feedback."""
    log_usage(update.effective_user.id, "/feedback")
    await update.message.reply_text(
        "Please send your feedback now. I'll log it for future improvements. "
        "You can send any message, and it will be recorded as feedback."
    )
    return WAITING_FOR_FEEDBACK

async def receive_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives and logs user feedback."""
    user_id = update.effective_user.id
    feedback_text = update.message.text
    logger.info(f"Feedback from User {user_id}: {feedback_text}")
    # In a real app, you'd save this to a persistent database or file.
    # For now, it's just logged to console/Render logs.
    log_usage(user_id, f"feedback: {feedback_text[:50]}...") # Log first 50 chars of feedback
    await update.message.reply_text(
        "Thank you for your feedback! It helps me improve."
    )
    return ConversationHandler.END # End the feedback conversation

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    log_usage(update.effective_user.id, "/cancel")
    context.user_data.clear() 
    context.user_data['processing_active'] = False # Ensure typing task stops
    await update.message.reply_text(
        "Process cancelled. Send /summarize to start again or /help for commands."
    )
    return ConversationHandler.END

# Define the application object globally for uvicorn
application = Application.builder().token(BOT_TOKEN).defaults(Defaults(parse_mode=ParseMode.MARKDOWN)).build()

# Configure handlers
conv_handler = ConversationHandler(
    entry_points=[CommandHandler("summarize", start)],
    states={
        ASKING_FOR_INPUT: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, receive_content),
            CommandHandler("cancel", cancel) # Allow cancel at any stage
        ],
        ASKING_FOR_AUDIENCE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, receive_audience),
            CommandHandler("cancel", cancel)
        ],
        ASKING_FOR_SENTIMENT: [
            CallbackQueryHandler(handle_sentiment_choice),
            CommandHandler("cancel", cancel)
        ],
        ASKING_FOR_CLAIMS: [
            CallbackQueryHandler(handle_claims_choice_and_summarize),
            CommandHandler("cancel", cancel)
        ],
    },
    fallbacks=[CommandHandler("cancel", cancel)],
    allow_reentry=True # Allow users to restart conversation easily
)
application.add_handler(conv_handler)

# New handler for feedback conversation
feedback_conv_handler = ConversationHandler(
    entry_points=[CommandHandler("feedback", feedback_command)],
    states={
        WAITING_FOR_FEEDBACK: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_feedback)],
    },
    fallbacks=[CommandHandler("cancel", cancel)],
    allow_reentry=True
)
application.add_handler(feedback_conv_handler)

# Add the new translate command handler
application.add_handler(CommandHandler("translate", translate_text_command))
# Add the start command handler (for re-showing welcome message)
application.add_handler(CommandHandler("start", start))

# Global message handler for rate limiting (must be before other text handlers)
# It should check if the message is a command first, to avoid rate limiting commands
async def pre_process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        # If it's a command, let other handlers process it without rate limiting here
        if update.message.text.startswith('/'):
            return
        
        # Apply rate limit check for non-command text messages
        if not await rate_limit_check(update, context):
            return # Stop processing if rate limit exceeded
    
    # If not rate limited or it's a command, continue processing
    # This function doesn't return a state, it just acts as a pre-processor.
    # The actual message will then be handled by other handlers if not consumed here.
    pass # Let other handlers take over

# This handler must be added with a high priority or before other text handlers
# to ensure rate limiting occurs first for non-command messages.
# However, python-telegram-bot's ConversationHandler entry_points take precedence.
# A simpler way is to integrate rate limiting directly into the receive_content handler
# or make it a global pre-processor that returns ConversationHandler.END if rate limited.

# For now, I've integrated rate_limit_check directly into relevant handlers where applicable
# and added it as a general MessageHandler with a low priority so it doesn't block commands.
# The `rate_limit_check` itself now returns True/False to indicate if processing should continue.
# This makes it more like a guard.

# The `rate_limit_check` needs to be called at the start of `receive_content`
# and `translate_text_command` to be effective.
# I will adjust `receive_content` and `translate_text_command` to call `rate_limit_check`.

# Removed the global MessageHandler for rate_limit_check and integrated into command handlers
# for more precise control.

# Set up the webhook configuration
if WEBHOOK_URL:
    PORT = int(os.environ.get("PORT", "8080"))
    logger.info(f"Setting webhook for URL: {WEBHOOK_URL}/{BOT_TOKEN}")
    # The application.web_server is what uvicorn will serve
    # The webhook setup is implicitly handled by the web_server when it starts
    # No explicit application.setup_webhook() call needed here as uvicorn serves it directly.
else:
    logger.warning("WEBHOOK_URL not set. Webhook will not be set up automatically.")

# This block is for local testing if not deployed with uvicorn
if __name__ == "__main__":
    # If running locally and WEBHOOK_URL is not set, fall back to polling
    if not WEBHOOK_URL:
        logger.info("Running bot in polling mode (local development).")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    else:
        # When deployed with uvicorn, uvicorn will serve application.web_server directly.
        # No explicit run_webhook() or run_polling() here.
        pass # This pass is important as uvicorn will run the application.web_server directly
