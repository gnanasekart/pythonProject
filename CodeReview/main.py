import logging
import os

from flask import Flask, request, send_from_directory
from github_api import get_pull_request_details, post_review_comment
from llm_integration import analyze_code
from rag import fetch_contextual_data
from orchestration import detect_language, should_review, create_prompt, get_or_store_fixture

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route('/')
def home():
    return 'Welcome to the Webhook Handler!'

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/webhook', methods=['POST'])
def handle_webhook():

    try:
        event = request.json

        logging.info("Received webhook event: %s", event)
        # Extract pull request details
        pr_details = get_pull_request_details(event)
        logging.info("Pull Request Details: PR Number: %s, Repository: %s, Files Changed: %s",
                     pr_details['pr_number'], pr_details['repository'], pr_details['files_changed'])

        # Detect language
        language = detect_language(pr_details['code'])
        logging.info("Detected Language: %s", language)

        # Determine if review is needed
        if should_review(language, pr_details['files_changed']):
            logging.info("Review required for PR %s", pr_details['pr_number'])

            # Fetch contextual data
            context = fetch_contextual_data(pr_details)
            logging.info("Fetched Contextual Data")

            # Create prompt for LLM
            prompt = create_prompt(pr_details['code'], context)
            logging.info("Created Prompt for LLM")

            # Analyze code with LLM
            analysis = analyze_code(prompt)
            logging.info("LLM Analysis: %s", analysis)

            # Ensure consistent analysis
            consistent_analysis = get_or_store_fixture(pr_details['code'], analysis)
            logging.info("Consistent Analysis: %s", consistent_analysis)

            # Post review comments to GitHub
            post_review_comment(pr_details['pr_number'], consistent_analysis)
            logging.info("Successfully posted comments to PR %s", pr_details['pr_number'])
        else:
            logging.info("No review needed for PR %s", pr_details['pr_number'])

    except Exception as e:
        logging.error("Error processing webhook: %s",  e, exc_info=True)
        return str(e), 500

    return '', 200

if __name__ == '__main__':
    # Start the Flask server
    app.run(port=5000, debug=True)