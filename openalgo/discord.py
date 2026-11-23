"""
OpenAlgo REST API Documentation - Discord Notification Methods
    https://docs.openalgo.in
"""

from .base import BaseAPI


class DiscordAPI(BaseAPI):
    """
    Discord notification API methods for OpenAlgo.
    Inherits from the BaseAPI class.
    """

    def discord(self, message, *, wait_for_delivery=False):
        """
        Post a message to your Discord channel through OpenAlgo.

        OpenAlgo posts it through the Discord channel webhook saved on its
        /discord page. The webhook is the only way in: there is no bot to run.
        Messages post only while Discord is switched on at OpenAlgo's /alerts
        page; the event groups chosen there apply to the automatic order alerts,
        not to messages sent here.

        Parameters:
        - message (str): The text to post, at most 2000 characters (Discord's
                         limit). Discord markdown is kept. Mentions never ping
                         anybody: @everyone, @here, user and role mentions are
                         shown as text. Required.
        - wait_for_delivery (bool, optional): False (default) queues the message
                         and returns at once. True posts it first and reports
                         whether Discord accepted it.

        The SDK sends the message as given and leaves the checks to the server,
        as telegram() and whatsapp() do: an empty message, one over 2000
        characters or a wait_for_delivery that is not True or False comes back
        as the server's 400 answer naming the field.

        Returns:
        dict: JSON response.
            Queued (default):
                {"status": "success", "message": "Notification queued."}
            Delivered (wait_for_delivery=True):
                {"status": "success", "message": "Message sent to Discord."}

        Errors come back as the server's JSON body with the HTTP status in "code":
        - 400: missing or empty message, a message over 2000 characters, or
          wait_for_delivery not true or false. The message names the field.
        - 401: "Invalid or missing API key".
        - 409: Discord is switched off on /alerts, or no Discord webhook is saved.
        - 429: more than 30 requests in a minute.
        - 502: with wait_for_delivery, Discord refused the message or could not
          be reached. The message says why and what to do.
        - 503: too many messages are already waiting to be sent.

        Rate Limiting:
        - 30 requests per minute, about what Discord accepts on one webhook. A
          failed post is not retried.

        Examples:
            # Queue a message and return at once
            api.discord("**NIFTY breakout** above 24,200")

            # Post now and learn whether Discord accepted it
            api.discord("Stop-loss hit on BANKNIFTY", wait_for_delivery=True)
        """
        payload = {
            "apikey": self.api_key,
            "message": message,
            "wait_for_delivery": wait_for_delivery,
        }
        return self._post("discord/notify", payload)
