#include "pipelineUtils.h"


GstElement *PipelineUtils::create_source_bin(guint index, gchar *uri) {
    GstElement *bin, *uri_decode_bin;
    gchar bin_name[16] = {};

    SPDLOG_INFO(bin_name,
                15, "source-bin-%02d", index);
    /* Create a source GstBin to abstract this bin's content from the rest of the
     * pipeline */
    bin = gst_bin_new(bin_name);

    /* Source element for reading from the uri.
     * decodebin and let it figure out the container format of the
     * stream and the codec and plug the appropriate demux and decode plugins. */
    uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

    if (!bin || !uri_decode_bin) {
        SPDLOG_ERROR("One element in source bin could not be created.");
        return nullptr;
    }

    /* Set the input uri to the source element */
    g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

    /* Connect to the "pad-added" signal of the decodebin which generates a
     * callback once a new pad for raw data has beed created by the decodebin */
    g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added", G_CALLBACK(cb_newpad),
                     bin);
    g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                     G_CALLBACK(decodebin_child_added), bin);

    gst_bin_add(GST_BIN(bin), uri_decode_bin);

    /* Create a ghost pad for the source bin which will act as a proxy
     * for the video decoder src pad. The ghost pad will not have a target right
     * now. Once the decode bin creates the video decoder and generates the
     * cb_newpad callback, we will set the ghost pad target to the video decoder
     * src pad. */
    if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src", GST_PAD_SRC))) {
        //SPDLOG_ERROR("Failed to add ghost pad in source bin");
        return nullptr;
    }
    return bin;
}

void PipelineUtils::cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data) {
    GstCaps *caps                      = gst_pad_get_current_caps(decoder_src_pad);
    const GstStructure *str            = gst_caps_get_structure(caps, 0);
    const gchar *name                  = gst_structure_get_name(str);
    auto *source_bin                   = (GstElement *) data;
    GstCapsFeatures *features          = gst_caps_get_features(caps, 0);
    const char *GST_CAPS_FEATURES_NVMM = "memory:NVMM";

    if (!strncmp(name, "video", 5)) {
        if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM)) {
            GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
            if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                          decoder_src_pad)) {
                SPDLOG_ERROR("Failed to link decoder src pad to source bin ghost pad");
            }
            gst_object_unref(bin_ghost_pad);
        } else {
            SPDLOG_ERROR("Decodebin did not pick nvidia decoder plugin.");
        }
    }
}

void PipelineUtils::decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                                          gchar *name, gpointer user_data) {
    SPDLOG_INFO("Decodebin child added: ", name);
    if (g_strrstr(name, "decodebin") == name) {
        g_signal_connect(G_OBJECT(object), "child-added",
                         G_CALLBACK(decodebin_child_added), user_data);
    }
    std::string name_str = std::string(name);

    if (name_str.find("nvv4l2decoder") != std::string::npos) {
        SPDLOG_INFO("Seting bufapi_version: {}", name);

        g_object_set(object, "bufapi-version", TRUE, NULL);
        g_object_set(object, "drop-frame-interval", 0, NULL);        
    }
}


gboolean PipelineUtils::bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    auto *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            SPDLOG_INFO("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            SPDLOG_ERROR("ERROR from element {}: {}",
                         GST_OBJECT_NAME(msg->src), error->message);
            if (debug)
                SPDLOG_ERROR("Error details: {}", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

guint64 PipelineUtils::shorten_track_id(guint64 track_id){
    return track_id % TRACK_ID_PRIME_NUM;
}