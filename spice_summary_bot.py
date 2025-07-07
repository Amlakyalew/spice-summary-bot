#!/usr/bin/env python3
import logging
import re
import requests
import json
import os # New import for environment variables
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

# --- Get bot token and API key from environment variables ---
# This is crucial for security and deployment on platforms like Render
BOT_TOKEN = os.environ.get("BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEBHOOK_URL = os.environ.get("RENDER_EXTERNAL_URL") # Render sets this env var

if not BOT_TOKEN:
    logging.error("BOT_TOKEN environment variable not set!")
    # In a real deployment, you might exit here. For local testing, you could set a default.
    pass 
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set!")
    # Same as above.
    pass
if not WEBHOOK_URL:
    logging.error("RENDER_EXTERNAL_URL environment variable not set! Webhook setup might fail.")


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# States for the conversation
ASKING_FOR_INPUT, ASKING_FOR_AUDIENCE, ASKING_FOR_SENTIMENT, ASKING_FOR_CLAIMS = range(4)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and asks the user for text or a URL to summarize."""
    welcome_message = (
        "Hello! I'm your Spice Summary Bot, designed to give you quick, tailored summaries of any text or URL. "
        "What makes me 'spicy'? I don't just summarize; I adapt to your needs!\\n\\n"
        "My key features include:\\n"
        "- *Customized Summaries:* Get summaries tailored for any audience (e.g., 'for a 10-year-old', 'for a technical expert', 'in simple terms').\\n"
        "- *Sentiment Analysis:* Optionally gauge the overall tone (positive, negative, or neutral) of the content.\\n"
        "- *Key Claims Extraction:* Easily identify the main arguments or claims within the text.\\n\\n"
        "Here are the commands you can use:\\n"
        "- /summarize: Start a new summarization process.\\n"
        "- /cancel: Stop any ongoing process.\\n"
        "- /start: See this welcome message again.\\n\\n"
        "Ready to get started? Send /summarize!"
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)
    return ASKING_FOR_INPUT

async def receive_content(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives text or a URL and stores it, then asks for audience."""
    user_input = update.message.text
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

    final_output = "Could not process the content."

    # Check if input is a URL
    url_pattern = re.compile(r"https?://(?:www\\.)?[\\w\\.-]+\\.\\w+(?:/[\\w\\.-]*)*(?:[\\?&][^=&]+=[^=&]+)*")
    
    text_content = ""
    if url_pattern.match(content): # Use stored content here
        try:
            response = requests.get(content, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            text_content = ' '.join([p.get_text() for p in paragraphs]).strip()

            if not text_content:
                text_content = soup.body.get_text(separator=' ', strip=True)
            
            if not text_content:
                final_output = "Could not extract readable text from the provided URL."
                await query.edit_message_text(final_output)
                return ConversationHandler.END

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {content}: {e}")
            final_output = f"Sorry, I couldn't fetch content from that URL. Please check the link or try another one. Error: {e}"
        except Exception as e:
            logger.error(f"Error processing URL content: {e}")
            final_output = "An unexpected error occurred while processing the URL content."
    else:
        text_content = content # Direct text input

    if text_content and len(text_content) > 0:
        if len(text_content) > 4000: # Limit input to LLM for summarization
            text_content = text_content[:4000]
            
        prompt_parts = [f"Summarize the following text for {audience} concisely."]
        
        if include_sentiment:
            prompt_parts.append("Also, analyze the overall sentiment (Positive, Negative, Neutral) and provide a brief reason for it.")
        
        if include_claims:
            prompt_parts.append("Additionally, extract 3-5 key claims or arguments from the text and list them as bullet points.")
        
        prompt_parts.append("Format the output as follows: 'Summary: [summary text]\\n[Optional Sentiment: Sentiment: [sentiment] (Reason: [reason])]\\n[Optional Claims: Key Claims:\\n- Claim 1\\n- Claim 2]'")
        prompt_parts.append(f":\\n\\n{text_content}")

        full_prompt = " ".join(prompt_parts)
        
        gemini_response = await call_gemini_api_from_python(full_prompt)
        final_output = gemini_response # Gemini will return the formatted string

    await query.edit_message_text(final_output)
    return ConversationHandler.END

async def call_gemini_api_from_python(prompt: str) -> str:
    """Calls the Gemini API to get a text generation response using Python's requests library."""
    apiKey = GEMINI_API_KEY # Use environment variable here

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

    try:
        full_api_url = apiUrl
        if apiKey:
            full_api_url += f"?key={apiKey}"

        response = requests.post(full_api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            logger.error(f"Unexpected API response structure from Gemini: {json.dumps(result)}")
            return "Sorry, the AI model did not return a valid summary."
    except requests.exceptions.RequestException as e:
        logger.error(f"Network or HTTP error calling Gemini API: {e}")
        return "Sorry, I couldn't connect to the AI model. Please check your internet connection and API key configuration."
    except Exception as e:
        logger.error(f"General error calling Gemini API: {e}")
        return "Sorry, an unexpected error occurred while getting the summary."

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    context.user_data.clear() 
    await update.message.reply_text(
        "Summarization process cancelled. Send /summarize to start again."
    )
    return ConversationHandler.END

# Define the application object globally for uvicorn
application = Application.builder().token(BOT_TOKEN).build()

# Configure handlers (moved outside main to ensure they are added when application is created)
conv_handler = ConversationHandler(
    entry_points=[CommandHandler("summarize", start)],
    states={
        ASKING_FOR_INPUT: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, receive_content),
        ],
        ASKING_FOR_AUDIENCE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, receive_audience),
        ],
        ASKING_FOR_SENTIMENT: [
            CallbackQueryHandler(handle_sentiment_choice),
        ],
        ASKING_FOR_CLAIMS: [
            CallbackQueryHandler(handle_claims_choice_and_summarize),
        ],
    },
    fallbacks=[CommandHandler("cancel", cancel)],
)
application.add_handler(conv_handler)

# Set up the webhook if WEBHOOK_URL is available
# This part is now handled directly by the application.web_server when uvicorn runs it.
if WEBHOOK_URL:
    PORT = int(os.environ.get("PORT", "8080")) # Render sets the PORT env var
    logger.info(f"Setting webhook for URL: {WEBHOOK_URL}/{BOT_TOKEN}")
    # The setup_webhook call is now part of the web_server's lifecycle when served by uvicorn
    # application.setup_webhook(listen="0.0.0.0", port=PORT, url_path=BOT_TOKEN, webhook_url=f"{WEBHOOK_URL}/{BOT_TOKEN}")
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
