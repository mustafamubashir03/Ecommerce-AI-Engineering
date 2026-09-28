"""The providers this application can talk to, one module each.

`openrouter` owns the model pool. The others are fallbacks, each optional and
each independent: a missing key or a disabled block means that provider is not
in the rotation, never a startup failure. Nothing here knows about the order
they are tried in, which is `routing.policy`'s job.
"""

from api.agents.providers import google, groq, openrouter

__all__ = ["google", "groq", "openrouter"]
