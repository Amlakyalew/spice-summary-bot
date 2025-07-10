import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables (secure token storage)
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")  # ✅ No hardcoded token!

# Initialize bot
app = ApplicationBuilder().token(TOKEN).build()

# Command: /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to Spice Summary Bot! Send me a PDF, and I'll summarize it for you."
    )

# Command: /generate (PDF processing)
async def generate_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message.document:
            await update.message.reply_text("Please send a PDF file.")
            return

        file = await update.message.document.get_file()
        pdf_path = f"temp_{update.message.document.file_id}.pdf"
        await file.download_to_drive(pdf_path)

        # Extract text from PDF
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages])

        # Simulate summary (replace with your logic)
        summary = f"📄 Summary (first 500 chars):\n{text[:500]}..."

        await update.message.reply_text(summary)
        os.remove(pdf_path)  # Clean up

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# Register handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.Document.PDF, generate_summary))

# Start bot (use polling for testing, webhook for hosting)
if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()  # 🚀 Replace with webhook in production
