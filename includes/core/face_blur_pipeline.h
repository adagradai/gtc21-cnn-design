#ifndef __DS_PIPELINE_H__
#define __DS_PIPELINE_H__

#include <glib.h>
#include <gst/gst.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/cudafilters.hpp"

#include <cstring>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "nvdsmeta.h"

#include "config_parser.h"
#include "pipelineUtils.h"
#include "nvtx3/nvToolsExt.h"

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem; 

class FaceBlurPipeline{
private:
	//Define Gstreamer loop
    GMainLoop *loop{};

    //Define Gstreamer elements
    GstElement *pipeline{}, *streammux{}, *pgie{}, *tracker{}, *convertor{}, *convertor2{},
               *nvdsosd{}, *videosink{}, *encoder{}, *h264parser{}, *muxer{}, *filesink{};
    guint                            bus_watch_id{};
    GstBus *bus{};
    
    nvtxDomainHandle_t nvtx_domain;
    cv::Ptr<cv::cuda::Filter> gpu_gaussian_filter;
    JsonConfig config;

    // adds source uri at particular index
    bool addSource(const std::string &source_uri, int source_index);
    
public:
	FaceBlurPipeline(JsonConfig parsedConfig):config(parsedConfig){		
        // initialization of nvtx domain
        std::string nvtx_str = "FaceBlurPipeline:";
        auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
        std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
        nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);
        nvtx_domain = nvtx_domain_ptr.release ();

        //initialization of gaussian filter ptr
        gpu_gaussian_filter = cv::cuda::createGaussianFilter(CV_8UC4, CV_8UC4, cv::Size(31,31), 10);
	}

    // create pipeline elements
    bool createElements();

    // sets elements properties
	bool setElementProperties();

    // links pipeline elemenets
    bool linkElements();

    // starts pipeline
    bool start();

    // stop pipeline 
    bool stop();

    // probe to blur out faces in pipeline
	static GstPadProbeReturn blurFacesInFrameProbe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);
};

#endif