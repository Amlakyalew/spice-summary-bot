import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# ======================
#  SETUP & CONFIGURATION
# ======================
# Initialize logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Validate token
if not TOKEN:
    logger.error("❌ TELEGRAM_TOKEN not found in environment variables!")
    exit(1)
logger.info(f"✅ Token loaded ({len(TOKEN)} characters)")

# ======================
#  BOT FUNCTIONALITY
# ======================
app = ApplicationBuilder().token(TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /start command"""
    await update.message.reply_text(
        "✨ Welcome to Spice Summary Bot!\n\n"
        "Send me a PDF file, and I'll extract text from it."
    )

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process PDF files"""
    try:
        if not update.message.document:
            await update.message.reply_text("⚠️ Please send a PDF file.")
            return

        # Download PDF
        file = await update.message.document.get_file()
        pdf_path = f"temp_{update.message.document.file_id}.pdf"
        await file.download_to_drive(pdf_path)
        logger.info(f"Downloaded PDF: {pdf_path}")

        # Extract text
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join(
                page.extract_text() 
                for page in reader.pages 
                if page.extract_text()
            )

        # Generate summary
        summary = (
            f"📄 Extracted Text (first 500 characters):\n\n"
            f"{text[:500]}..." if text 
            else "❌ No text could be extracted from this PDF."
        )

        await update.message.reply_text(summary)
        os.remove(pdf_path)  # Clean up
        logger.info("PDF processed successfully")

    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

# ======================
#  MAIN EXECUTION
# ======================
# Register handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

if __name__ == "__main__":
    logger.info("🚀 Starting bot...")
    app.run_polling()
