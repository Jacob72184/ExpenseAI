import base64
import logging
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)

def send_token_email(session_credentials_dict, recipient_address, email_subject, secure_token):
    """
    Sends an automated HTML email containing a secure token through the Gmail API.

    Parameters:
    - session_credentials_dict (dict): OAuth2 tokens stored in your session state.
    - recipient_address (str): The email address of the target recipient.
    - email_subject (str): The subject line header of the email.
    - secure_token (str): The specific token string you want to send.
    """
    try:
        # Reconstruct the authenticated Google credentials from the session dictionary
        creds = Credentials(**session_credentials_dict)
        service = build('gmail', 'v1', credentials=creds)

        # Clean HTML layout focusing entirely on displaying your token string
        html_body = f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333; padding: 20px;">
            <div style="max-width: 550px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px;">
              <h2 style="color: #1a73e8; margin-top: 0;">Your Secure Token</h2>
              <p>Below is the secure access token required for your request:</p>
              <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #1a73e8; font-family: monospace; font-size: 16px; word-break: break-all; margin: 20px 0; color: #202124;">
                <strong>{secure_token}</strong>
              </div>
              <p style="font-size: 12px; color: #666666;">If you did not initiate this action, please disregard this email securely.</p>
            </div>
          </body>
        </html>
        """

        # Build the standard MIME envelope layout
        message = MIMEText(html_body, 'html')
        message['to'] = recipient_address
        message['subject'] = email_subject

        # Compile into the url-safe base64 text string required by Google
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        payload = {'raw': raw_message}

        # Dispatch through Google's gateway
        sent_message = service.users().messages().send(userId="me", body=payload).execute()
        return {"status": "success", "message_id": sent_message['id']}

    except Exception as e:
        # Never log secrets; the exception may contain credential fragments.
        logger.exception("Gmail send_token_email failed for %s", recipient_address)
        return {"status": "error", "details": "Could not send the invitation email."}
