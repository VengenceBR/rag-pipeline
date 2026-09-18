=== Corvit AI Assistant ===
Contributors: vengencebr
Tags: chatbot, ai, chat widget, rag, customer support
Requires at least: 5.8
Tested up to: 6.6
Requires PHP: 7.2
Stable tag: 1.0.0
License: GPLv2 or later
License URI: https://www.gnu.org/licenses/gpl-2.0.html

Adds a floating AI chat assistant to the site, backed by a hybrid retrieval-augmented
generation (RAG) system that answers visitor questions grounded in Corvit's own documents.

== Description ==

This plugin adds a single, minimal script tag to the public-facing site, which renders a
floating chat bubble. Clicking it opens a chat panel (embedded via an isolated iframe, so it
cannot conflict with the site's own CSS or JavaScript) where visitors can ask questions and get
answers grounded in Corvit's real documentation, with citations.

The chat backend itself is a separate service (see
https://github.com/VengenceBR/rag-pipeline) -- this plugin's only job is loading its widget
script on the site the WordPress-native way, instead of editing theme files directly.

= What this plugin does NOT do =

* It does not store any data in WordPress. All questions and answers are handled entirely by
  the external backend.
* It does not add any block, shortcode, or widget-area component -- the chat bubble is
  injected site-wide via `wp_footer`, matching how most chat-widget plugins work.
* It does not modify any theme files.

== Installation ==

1. Upload the `corvit-ai-assistant` folder to `/wp-content/plugins/`, or install the zip via
   Plugins > Add New > Upload Plugin.
2. Activate the plugin through the "Plugins" menu in WordPress.
3. Go to Settings > Corvit AI Assistant and enter the backend URL (where the RAG backend is
   deployed). The widget will not appear until this is set.

== Frequently Asked Questions ==

= Does this slow down page load? =

The widget script loads asynchronously (`async`) and only after the rest of the page, so it
does not block rendering.

= Can I turn it off without deactivating the plugin? =

Yes -- uncheck "Enable chat widget" on the settings page.

== Changelog ==

= 1.0.0 =
* Initial release.
