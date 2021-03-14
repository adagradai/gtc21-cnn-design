#ifndef __DS_PIPELINE_H__
#define __DS_PIPELINE_H__

#include <glib.h>
#include <gst/gst.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "pipelineUtils.h"
#include <cstring>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "nvdsmeta.h"
#include "config_parser.h"

class FaceBlurPipeline{
private:
	//Define Gstreamer loop
    GMainLoop *loop{};
    //Define Gstreamer elements
    GstElement *pipeline{}, *streammux{}, *pgie{}, *tracker,
               *tiler{}, *convertor{},
               *nvdsosd{}, *videosink{}, *fake_sink{};
    guint                            bus_watch_id{};
    GstBus *bus{};
    
    JsonConfig config;

    static NvBufSurface              *inter_buf;
    static cudaStream_t              cuda_stream;
    static bool init_object_crop_settings(uint muxer_height, uint muxer_width);
    static bool get_rgb_frame(NvBufSurface *input_buf, FrameCropDims &crop_rect_params, int frame_idx, cv::Mat &rgbMat);    
    bool addSource(const std::string &source_uri, int source_index);
public:
	FaceBlurPipeline(JsonConfig parsedConfig):config(parsedConfig){		
	}
	bool setElementProperties();
    bool createElements();
    bool linkElements();
    bool start();
    bool stop(); 
    ~FaceBlurPipeline();   
	static GstPadProbeReturn pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);
};

#endif