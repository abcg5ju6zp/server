"""Asynchronous batch player commands controller package.

The controller class is imported from ``.controller`` explicitly; this package init
stays empty so that importing a standalone submodule (e.g. the conflict error from the
webserver layer) does not pull the player controller chain in during early startup.
"""
