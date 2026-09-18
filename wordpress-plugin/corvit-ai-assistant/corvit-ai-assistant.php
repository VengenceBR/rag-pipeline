<?php
/**
 * Plugin Name: Corvit AI Assistant
 * Plugin URI: https://github.com/VengenceBR/rag-pipeline
 * Description: Adds the Corvit AI Assistant chat widget (a floating chat bubble backed by a hybrid retrieval-augmented generation system, answering visitor questions from Corvit's own documents) to the site. Loads a single external script; no other site files are modified.
 * Version: 1.0.0
 * Requires at least: 5.8
 * Requires PHP: 7.2
 * Author: Muhammad Abdullah
 * License: GPL v2 or later
 * License URI: https://www.gnu.org/licenses/gpl-2.0.html
 * Text Domain: corvit-ai-assistant
 */

// Block direct access to this file.
if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

define( 'CORVIT_AI_ASSISTANT_VERSION', '1.0.0' );
define( 'CORVIT_AI_ASSISTANT_OPTION_GROUP', 'corvit_ai_assistant_settings' );
define( 'CORVIT_AI_ASSISTANT_URL_OPTION', 'corvit_ai_assistant_backend_url' );
define( 'CORVIT_AI_ASSISTANT_ENABLED_OPTION', 'corvit_ai_assistant_enabled' );

/**
 * The backend URL is required and has no built-in default -- it must be set
 * once via Settings > Corvit AI Assistant, pointing at wherever the RAG
 * backend (github.com/VengenceBR/rag-pipeline) is deployed. Until it's set,
 * the widget script is not enqueued at all, so an unconfigured install never
 * tries to load a broken/placeholder URL on the live site.
 */
function corvit_ai_assistant_get_backend_url() {
	$url = get_option( CORVIT_AI_ASSISTANT_URL_OPTION, '' );
	return is_string( $url ) ? untrailingslashit( trim( $url ) ) : '';
}

/**
 * Enqueues the widget loader on the public-facing site only. The loader
 * script itself (served by the backend, not this plugin) renders the bubble
 * and iframe -- this plugin's only job is getting that one script tag onto
 * the page, the WordPress-native way, instead of a raw theme-file edit.
 */
function corvit_ai_assistant_enqueue_script() {
	if ( is_admin() ) {
		return;
	}

	$enabled = get_option( CORVIT_AI_ASSISTANT_ENABLED_OPTION, '1' );
	if ( '1' !== $enabled ) {
		return;
	}

	$backend_url = corvit_ai_assistant_get_backend_url();
	if ( empty( $backend_url ) ) {
		return;
	}

	wp_enqueue_script(
		'corvit-ai-assistant-widget',
		$backend_url . '/widget-loader.js',
		array(),
		CORVIT_AI_ASSISTANT_VERSION,
		true
	);

	// The loader script derives its own origin from its <script src>, so no
	// inline configuration is needed here -- this just adds async loading so
	// the widget never blocks the rest of the page from rendering.
	add_filter(
		'script_loader_tag',
		function ( $tag, $handle ) {
			if ( 'corvit-ai-assistant-widget' === $handle ) {
				return str_replace( ' src', ' async src', $tag );
			}
			return $tag;
		},
		10,
		2
	);
}
add_action( 'wp_enqueue_scripts', 'corvit_ai_assistant_enqueue_script' );

/**
 * Settings: Settings > Corvit AI Assistant.
 */
function corvit_ai_assistant_register_settings() {
	register_setting(
		CORVIT_AI_ASSISTANT_OPTION_GROUP,
		CORVIT_AI_ASSISTANT_URL_OPTION,
		array(
			'type'              => 'string',
			'sanitize_callback' => 'esc_url_raw',
			'default'           => '',
		)
	);
	register_setting(
		CORVIT_AI_ASSISTANT_OPTION_GROUP,
		CORVIT_AI_ASSISTANT_ENABLED_OPTION,
		array(
			'type'              => 'string',
			'sanitize_callback' => 'sanitize_text_field',
			'default'           => '1',
		)
	);
}
add_action( 'admin_init', 'corvit_ai_assistant_register_settings' );

function corvit_ai_assistant_add_settings_page() {
	add_options_page(
		'Corvit AI Assistant',
		'Corvit AI Assistant',
		'manage_options',
		'corvit-ai-assistant',
		'corvit_ai_assistant_render_settings_page'
	);
}
add_action( 'admin_menu', 'corvit_ai_assistant_add_settings_page' );

function corvit_ai_assistant_render_settings_page() {
	if ( ! current_user_can( 'manage_options' ) ) {
		return;
	}

	$backend_url = corvit_ai_assistant_get_backend_url();
	$enabled     = get_option( CORVIT_AI_ASSISTANT_ENABLED_OPTION, '1' );
	?>
	<div class="wrap">
		<h1>Corvit AI Assistant</h1>

		<?php if ( empty( $backend_url ) ) : ?>
			<div class="notice notice-warning">
				<p><strong>Not active yet</strong> -- set the backend URL below. The chat widget
				will not appear on the site until this is configured.</p>
			</div>
		<?php endif; ?>

		<form method="post" action="options.php">
			<?php settings_fields( CORVIT_AI_ASSISTANT_OPTION_GROUP ); ?>
			<table class="form-table" role="presentation">
				<tr>
					<th scope="row">
						<label for="corvit_ai_assistant_enabled">Enable chat widget</label>
					</th>
					<td>
						<input type="hidden" name="<?php echo esc_attr( CORVIT_AI_ASSISTANT_ENABLED_OPTION ); ?>" value="0" />
						<label>
							<input
								type="checkbox"
								id="corvit_ai_assistant_enabled"
								name="<?php echo esc_attr( CORVIT_AI_ASSISTANT_ENABLED_OPTION ); ?>"
								value="1"
								<?php checked( $enabled, '1' ); ?>
							/>
							Show the chat bubble on the site
						</label>
					</td>
				</tr>
				<tr>
					<th scope="row">
						<label for="corvit_ai_assistant_backend_url">Backend URL</label>
					</th>
					<td>
						<input
							type="url"
							id="corvit_ai_assistant_backend_url"
							name="<?php echo esc_attr( CORVIT_AI_ASSISTANT_URL_OPTION ); ?>"
							value="<?php echo esc_attr( $backend_url ); ?>"
							class="regular-text"
							placeholder="https://your-backend-domain.example.com"
						/>
						<p class="description">
							Where the Corvit AI Assistant backend
							(<a href="https://github.com/VengenceBR/rag-pipeline" target="_blank" rel="noopener noreferrer">rag-pipeline</a>)
							is deployed. No trailing slash.
						</p>
					</td>
				</tr>
			</table>
			<?php submit_button(); ?>
		</form>
	</div>
	<?php
}
