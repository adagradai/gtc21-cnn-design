#include "face_blur_pipeline.h"

/* initialization of static members */
NvBufSurface * FaceBlurPipeline::inter_buf        = nullptr;
cudaStream_t FaceBlurPipeline::cuda_stream;

bool FaceBlurPipeline::addSource(const std::string &source_uri, int source_index) {
    GstPad *sinkpad, *srcpad;

    GstElement *source_bin = PipelineUtils::create_source_bin(source_index, (char *) source_uri.c_str());
    if (!source_bin) {
        SPDLOG_ERROR("Can't create source");
        return false;
    } else {
        SPDLOG_INFO("Create source");
    }

    gst_bin_add(GST_BIN(pipeline), source_bin);

    std::string padName = "sink_" + std::to_string(source_index);
    sinkpad             = gst_element_get_request_pad(streammux, padName.c_str());
    if (!sinkpad) {
        SPDLOG_ERROR("Can't create sinkpad");
        return false;
    } else {
        SPDLOG_INFO("Created sinkpad");
    }
    srcpad = gst_element_get_static_pad(source_bin, "src");
    if (!srcpad) {
        SPDLOG_ERROR("No srcpad!");
        return false;
    } else {
        SPDLOG_INFO("Created srcpad");
    }
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        SPDLOG_INFO("can't link!!");
        return false;
    } else {
        SPDLOG_INFO("Linked properly");
    }

    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);

    //set properties of source meta for ith source
    return true;
}


//generates video consisting of frames based on track id
GstPadProbeReturn FaceBlurPipeline::pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data){
    auto *buf = (GstBuffer *) info->data;
    NvDsObjectMeta *obj_meta;
    NvDsMetaList *l_frame;
    NvDsMetaList *l_obj;
    
     //Get original raw data
    GstMapInfo in_map_info;

    FaceBlurPipeline * recorder = (FaceBlurPipeline *) u_data;
    if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ)) {
        SPDLOG_ERROR("Error: Failed to map gst buffer");
        gst_buffer_unmap(buf, &in_map_info);
        return GST_PAD_PROBE_OK;
    }
    
    SPDLOG_INFO("pgie src pad buffer probe hit");
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    // NV12 to RGB conversion
    NvBufSurface *input_surface = (NvBufSurface *) in_map_info.data;
    
    int frame_index =0;
    for (l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next) {        
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);                        
        for (l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);

            //check if class id of object is face
            if(obj_meta->class_id ==0){                                
                FrameCropDims cropDims;
                cropDims.x= 0;
                cropDims.y = 0;
                cropDims.w = FaceBlurPipeline::inter_buf->surfaceList[0].width;
                cropDims.h = FaceBlurPipeline::inter_buf->surfaceList[0].height;

                cv::Mat rgbMat = cv::Mat(cv::Size(FaceBlurPipeline::inter_buf->surfaceList[0].width,
                                                  FaceBlurPipeline::inter_buf->surfaceList[0].height),
                                                  CV_8UC3);
                bool conversionStatus = FaceBlurPipeline::get_rgb_frame(input_surface, cropDims, frame_index, rgbMat);                                
                if(conversionStatus){
                    //guint64 short_track_id = PipelineUtils::shorten_track_id(obj_meta->object_id);
                    //recorder->addFrameToVideo(short_track_id, rgbMat);
                    //SPDLOG_INFO("saving rgb mat to video");
                    SPDLOG_INFO("frame rgb conversion successful");
                }
                else
                    SPDLOG_INFO("failed to convert rgb frame!!");
            }
        }
        frame_index +=1 ;
    }

    gst_buffer_unmap(buf, &in_map_info);
    return GST_PAD_PROBE_OK;
}

bool FaceBlurPipeline::createElements() {
    loop          = g_main_loop_new(nullptr, FALSE);
    pipeline      = gst_pipeline_new("BusVideoRecorder");    
    streammux     = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie          = gst_element_factory_make("nvinfer", "primary-nvinference-engine"); 	  
    

    // enable output streaming 
    if(config.stream_output){        
        tracker       = gst_element_factory_make("nvtracker", "tracker");
        tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
        convertor = gst_element_factory_make ("nvvideoconvert", "convertor");
        nvdsosd = gst_element_factory_make("nvdsosd", "nvdsosd");
        videosink = gst_element_factory_make ("nveglglessink", "nveglglessink");
    }
    else
        fake_sink     = gst_element_factory_make("fakesink", "fakesink");

    //Add streammux and sources
    gst_bin_add(GST_BIN(pipeline), streammux);
    bool sourceAdditionStatus = FaceBlurPipeline::addSource(config.camera_uri, 0);
    if (!sourceAdditionStatus) {
        SPDLOG_ERROR("Source not added properly");
        return false;
    }

    //Check if elements created
    if (!loop || !pipeline || !streammux || !pgie) {
        SPDLOG_ERROR("Could not create primary elements");
        return false;
    }
    if(config.stream_output && (!tracker || !tiler || !convertor || !nvdsosd || !videosink)){
        SPDLOG_ERROR("Could not create inference elements");
        return false;
    }
    else if(!config.stream_output && !fake_sink){
        SPDLOG_ERROR("Could not create fake sink");
        return false;
    }        
    return true;
}

bool FaceBlurPipeline::linkElements() {    
    if(config.stream_output){
        gst_bin_add_many(GST_BIN(pipeline), pgie, tracker, tiler, convertor, nvdsosd, videosink, nullptr);
        if (!gst_element_link_many(streammux, pgie, tracker, tiler, convertor, nvdsosd, videosink, NULL)) {
            SPDLOG_ERROR("Could not link one or more elements in case of streaming enabled");
            return false;
        }
        return true;
    }
    else{
        gst_bin_add_many(GST_BIN(pipeline), pgie, fake_sink, nullptr);
        if (!gst_element_link_many(streammux, pgie, fake_sink, NULL)) {
            SPDLOG_ERROR("Could not link one or more elements in case of streaming disabled");
            return false;
        }
        return true;
    }
}

bool FaceBlurPipeline::setElementProperties() {
    //Init crop settings
    if(!FaceBlurPipeline::init_object_crop_settings(config.muxer_height, config.muxer_width)){
        SPDLOG_INFO("failed to initialize intermediate rgb mat");
        return false;
    }
	
    GstPad * pgie_sink_pad = gst_element_get_static_pad(pgie, "src");
    if (!pgie_sink_pad)
        SPDLOG_ERROR("Unable to get sink pad\n");
    else
        gst_pad_add_probe(pgie_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_src_pad_buffer_probe, (gpointer) this, nullptr);
    
    //Streammux properties
    g_object_set(G_OBJECT(streammux),
                 "width", config.muxer_width,
                 "height", config.muxer_height,
                 "batched-push-timeout", 40000,
                 "batch-size", 1,
                 "live-source", 1,
                 NULL);

    //GIE properties
    g_object_set(G_OBJECT(pgie),
				 "config-file-path",
				  config.model_config_path.c_str(),
				  NULL);    	

    if(config.stream_output){
        //set properties of tracker
         g_object_set(G_OBJECT(tracker),
                 "ll-lib-file", config.tracker_lib_path.c_str(),
                 "ll-config-file", config.tracker_config_path.c_str(),
                 "tracker-width", config.tracker_width,
                 "tracker-height", config.tracker_height, NULL);
    }

    return true;
}


bool FaceBlurPipeline::get_rgb_frame(NvBufSurface *input_buf, FrameCropDims &crop_rect_params, int frame_idx, cv::Mat &rgbMat) {
    //transform nv12 to rgba
    NvBufSurfTransform_Error err;
    NvBufSurfTransformConfigParams transform_config_params;
    NvBufSurfTransformParams transform_params;
    NvBufSurfTransformRect src_rect;
    NvBufSurfTransformRect dst_rect;
    NvBufSurface ip_surf;
    ip_surf           = *input_buf;
    ip_surf.numFilled = ip_surf.batchSize = 1;
    ip_surf.surfaceList                   = &(input_buf->surfaceList[frame_idx]);
    gint src_left                         = GST_ROUND_UP_2((unsigned int) crop_rect_params.x);
    gint src_top                          = GST_ROUND_UP_2((unsigned int) crop_rect_params.y);
    gint src_width                        = GST_ROUND_DOWN_2((unsigned int) crop_rect_params.w);
    gint src_height                       = GST_ROUND_DOWN_2((unsigned int) crop_rect_params.h);

    // Configure transform session parameters for the transformation
    transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
    transform_config_params.gpu_id       = input_buf->gpuId;         //process on same gpu on source stream
    transform_config_params.cuda_stream  = FaceBlurPipeline::cuda_stream;//cuda stream

    err = NvBufSurfTransformSetSessionParams(&transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
        SPDLOG_ERROR("NvBufSurfTransformSetSessionParams failed");
        return false;
    }

    // Set the transform ROIs for source and destination
    src_rect = {(guint) src_top, (guint) src_left, (guint) src_width, (guint) src_height};
    dst_rect = {(guint) src_top, (guint) src_left, (guint) src_width, (guint) src_height};

    transform_params.src_rect         = &src_rect;
    transform_params.dst_rect         = &dst_rect;
    transform_params.transform_flag   = NVBUFSURF_TRANSFORM_CROP_SRC;
    transform_params.transform_filter = NvBufSurfTransformInter_Default;
    
    //        NvBufSurfaceMemSet(ANPRPipeline::inter_buf, 0, 0, 0);
    err = NvBufSurfTransform(&ip_surf, FaceBlurPipeline::inter_buf, &transform_params);

    if (err != NvBufSurfTransformError_Success) {
        SPDLOG_ERROR("NvBufSurfTransform failed with error %d while converting buffer");
        return false;
    }

    // Map the buffer so that it can be accessed by CPU
    if (NvBufSurfaceMap(FaceBlurPipeline::inter_buf, 0, 0, NVBUF_MAP_READ) != 0) {
        return false;
    }

    NvBufSurfaceSyncForCpu(FaceBlurPipeline::inter_buf, 0, 0);
    cv::Mat *rgbaFrame = new cv::Mat(FaceBlurPipeline::inter_buf->surfaceList[0].height,
                                     FaceBlurPipeline::inter_buf->surfaceList[0].width,
                                     CV_8UC4,
                                     FaceBlurPipeline::inter_buf->surfaceList[0].mappedAddr.addr[0],
                                     FaceBlurPipeline::inter_buf->surfaceList[0].pitch);
#if (CV_MAJOR_VERSION >= 4)
    cv::cvtColor(*rgbaFrame, rgbMat, cv::COLOR_RGBA2BGR);
#else
    cv::cvtColor(*rgbaFrame, rgbMat, CV_RGBA2BGR);
#endif
    delete rgbaFrame;
    return true;
}

bool FaceBlurPipeline::init_object_crop_settings(uint muxer_height, uint muxer_width){
    NvBufSurfaceCreateParams create_params;
    create_params.gpuId       = 0;
    create_params.width       =  muxer_width;
    create_params.height      =  muxer_height;
    create_params.size        = 0;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.layout      = NVBUF_LAYOUT_PITCH;
    create_params.memType     = NVBUF_MEM_CUDA_UNIFIED;

    /* create intermediate surface */
    if (NvBufSurfaceCreate(&FaceBlurPipeline::inter_buf, 1, &create_params) != 0) {
        SPDLOG_ERROR("Error: Could not allocate internal buffer for dsexample");
        return false;
    }

    if (cudaStreamCreate(&FaceBlurPipeline::cuda_stream) != cudaSuccess) {
        SPDLOG_ERROR("Error while creating cuda stream");
        return false;
    }
    return true;
}

bool FaceBlurPipeline::start() {
    //Wait till pipeline encounters an error or EOS
    /* Wait till pipeline encounters an error or EOS */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, PipelineUtils::bus_call, loop);
    gst_object_unref(bus);
    GstState pipelineState;
    /* Set the pipeline to "playing" state */    
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    // GetStateChangeReturn to check Pipeline state has changed
    GstStateChangeReturn ret = gst_element_get_state(pipeline, &pipelineState, NULL, GST_CLOCK_TIME_NONE);
    if ((ret == GST_STATE_CHANGE_SUCCESS) && (pipelineState == GST_STATE_PLAYING)) {
        g_main_loop_run(loop);        
        return true;
    }
    else
        return false;
}

/// stop pipeline
/// \return the bool status of the execution of the method
bool FaceBlurPipeline::stop() {
    if (pipeline != nullptr && loop != nullptr) { 
        GstStateChangeReturn ret = gst_element_get_state(pipeline, NULL, NULL, GST_CLOCK_TIME_NONE);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        g_source_remove(bus_watch_id);
        g_main_loop_unref(loop);
        return (ret == GST_STATE_CHANGE_SUCCESS);
    }
    return true;
}

FaceBlurPipeline::~FaceBlurPipeline(){
	if(FaceBlurPipeline::inter_buf){
		NvBufSurfaceDestroy(FaceBlurPipeline::inter_buf);
	}
    cudaStreamDestroy(FaceBlurPipeline::cuda_stream);
}