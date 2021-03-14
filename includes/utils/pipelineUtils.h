#ifndef __PIPELINE_UTILS_H__
#define __PIPELINE_UTILS_H__

#include <gst/gst.h>
#include <string>
#include <spdlog/spdlog.h>

#define TRACK_ID_PRIME_NUM 99991

class PipelineUtils {

public:
    static GstElement *create_source_bin(guint index, gchar *uri);
    
    static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data);

    static void cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data);

    static void decodebin_child_added(GstChildProxy *child_proxy, GObject *object, gchar *name, gpointer user_data);    

    static guint64 shorten_track_id(guint64 track_id);
};

#endif