"""
OpenAlgo REST API Documentation - Slack Notification Methods
    https://docs.openalgo.in
"""

from .base import BaseAPI


class SlackAPI(BaseAPI):
    """
    Slack notification API methods for OpenAlgo.
    Inherits from the BaseAPI class.
    """

    def slack(self, message, *, wait_for_delivery=False):
        """
        Post a message to your Slack channel through OpenAlgo.

        OpenAlgo posts it through the Slack incoming webhook saved on its /slack
        page. The webhook is the only way in: there is no bot to run. Messages
        post only while Slack is switched on at OpenAlgo's /alerts page; the
        event groups chosen there apply to the automatic order alerts, not to
        messages sent here.

        Parameters:
        - message (str): The text to post, at most 4000 characters. Slack bold
                         (*text*) works. The server escapes '&', '<' and '>', so
                         mentions and link markup are shown as text; send the
                         message as written, not escaped. Required.
        - wait_for_delivery (bool, optional): False (default) queues the message
                         and returns at once. True posts it first and reports
                         whether Slack accepted it.

        The SDK sends the message as given and leaves the checks to the server,
        as telegram() and whatsapp() do: an empty message, one over 4000
        characters or a wait_for_delivery that is not True or False comes back
        as the server's 400 answer naming the field.

        Returns:
        dict: JSON response.
            Queued (default):
                {"status": "success", "message": "Notification queued."}
            Delivered (wait_for_delivery=True):
                {"status": "success", "message": "Message sent to Slack."}

        Errors come back as the server's JSON body with the HTTP status in "code":
        - 400: missing or empty message, a message over 4000 characters, or
          wait_for_delivery not true or false. The message names the field.
        - 401: "Invalid or missing API key".
        - 409: Slack is switched off on /alerts, or no Slack webhook is saved.
        - 429: more than 30 requests in a minute.
        - 502: with wait_for_delivery, Slack refused the message or could not be
          reached. The message says why and what to do.
        - 503: too many messages are already waiting to be sent.

        Rate Limiting:
        - 30 requests per minute, about what Slack accepts on one webhook. A
          failed post is not retried.

        Examples:
            # Queue a message and return at once
            api.slack("*NIFTY breakout* above 24,200")

            # Post now and learn whether Slack accepted it
            api.slack("Stop-loss hit on BANKNIFTY", wait_for_delivery=True)
        """
        payload = {
            "apikey": self.api_key,
            "message": message,
            "wait_for_delivery": wait_for_delivery,
        }
        return self._post("slack/notify", payload)
