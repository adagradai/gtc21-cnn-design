#include "face_blur_pipeline.h"

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
    return true;
}

GstPadProbeReturn FaceBlurPipeline::blurFacesInFrameProbe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data){
    FaceBlurPipeline * pipelineInst = (FaceBlurPipeline *) u_data;

    if(pipelineInst == NULL){
        SPDLOG_INFO("pipeline instance is not passed!!.. skipping rest of face blur operation");
        return GST_PAD_PROBE_OK;
    }
      
    auto *buf = (GstBuffer *) info->data;
    NvDsMetaList *l_frame;
    
    //Get original raw data
    GstMapInfo in_map_info;

    if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ)) {
        SPDLOG_ERROR("Error: Failed to map gst buffer");
        gst_buffer_unmap(buf, &in_map_info);
        return GST_PAD_PROBE_OK;
    }
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);    
    NvBufSurface *input_surface = (NvBufSurface *) in_map_info.data;
    
    if (input_surface->memType != NVBUF_MEM_CUDA_UNIFIED){
        SPDLOG_INFO("input surface memory type is not unified!!! Skipping processing over objects");
        gst_buffer_unmap(buf, &in_map_info);
        return GST_PAD_PROBE_OK;
    }
   
    // iterate over all frames of batch
    for (l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next) {        
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);                        
      
        // Map the surface so that it can be accessed by CPU
        if (NvBufSurfaceMap(input_surface, frame_meta->batch_id, 0, NVBUF_MAP_READ_WRITE) != 0) {
            SPDLOG_INFO("failed to map surface!!");
            continue;
        }

        //use gpu mat to point to in-memory surface                   
        cv::cuda::GpuMat gpu_mat = cv::cuda::GpuMat(input_surface->surfaceList[frame_meta->batch_id].planeParams.height[0],
                                                    input_surface->surfaceList[frame_meta->batch_id].planeParams.width[0], CV_8UC4,
                                                    input_surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
                                                    input_surface->surfaceList[frame_meta->batch_id].planeParams.pitch[0]);
        
        // check all objects in face and filter out only faces
        for (auto l_obj =frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next) {
            NvDsObjectMeta * obj_meta = (NvDsObjectMeta *) (l_obj->data);

            //check if class id of object is face
            if(obj_meta->class_id == 0){         
                cv::Rect face_crop_rect = cv::Rect(obj_meta->rect_params.left, 
                                            obj_meta->rect_params.top, 
                                            obj_meta->rect_params.width, 
                                            obj_meta->rect_params.height); 

                //push event attribute for gpu operation
                nvtxEventAttributes_t eventAttrib = {0};
                eventAttrib.version = NVTX_VERSION;
                eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                eventAttrib.colorType = NVTX_COLOR_ARGB;
                eventAttrib.color = 0x88DAF500;
                eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
                eventAttrib.message.ascii = "face blur operation GPU";
                nvtxDomainRangePushEx(pipelineInst->nvtx_domain, &eventAttrib);                                                                                                                                               

                // blur operation over detected face
                pipelineInst->gpu_gaussian_filter->apply(gpu_mat(face_crop_rect), gpu_mat(face_crop_rect));  

                // pop out event attribute
                nvtxDomainRangePop(pipelineInst->nvtx_domain);                                  
            }
        }
            
        // unmap frame surface post blur operation
        if (NvBufSurfaceUnMap (input_surface, frame_meta->batch_id, 0) !=0){
            SPDLOG_INFO("failed to unmap surface!! terminate");
            continue;
        }
    }

    gst_buffer_unmap(buf, &in_map_info);    
    return GST_PAD_PROBE_OK;
}

bool FaceBlurPipeline::createElements() {
    loop          = g_main_loop_new(nullptr, FALSE);
    pipeline      = gst_pipeline_new("DsFaceBlurPipeline");    
    streammux     = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie          = gst_element_factory_make("nvinfer", "primary-nvinference-engine"); 	  
    tracker       = gst_element_factory_make("nvtracker", "tracker");
    convertor     = gst_element_factory_make ("nvvideoconvert", "convertor");
    nvdsosd = gst_element_factory_make("nvdsosd", "nvdsosd");

    // enable output streaming 
    if(config.enable_display){        
        videosink = gst_element_factory_make ("nveglglessink", "nveglglessink");
    }
    else{
        convertor2 =  gst_element_factory_make ("nvvideoconvert", "h264parser2");
        encoder = gst_element_factory_make("nvv4l2h264enc", "nvv4l2h264-encoder");
        h264parser = gst_element_factory_make("h264parse", "h264-parser");
        muxer =  gst_element_factory_make ("qtmux", "muxer");        
        filesink = gst_element_factory_make("filesink", "filesink");
    }

    //Add streammux and sources
    gst_bin_add(GST_BIN(pipeline), streammux);
    bool sourceAdditionStatus = FaceBlurPipeline::addSource(config.source_uri, 0);
    if (!sourceAdditionStatus) {
        SPDLOG_ERROR("Source not added properly");
        return false;
    }

    //Check if elements created
    if (!loop || !pipeline || !streammux || !pgie || !tracker || !convertor || !nvdsosd) {
        SPDLOG_ERROR("Could not create primary elements");
        return false;
    }

    if(config.enable_display && !videosink){
        SPDLOG_ERROR("Could not create inference elements");
        return false;
    }
    else if(!config.enable_display && (!convertor2 || !encoder || !h264parser || !muxer || !filesink)){
        SPDLOG_ERROR("Could not create fake sink");
        return false;
    }        
    return true;
}

bool FaceBlurPipeline::setElementProperties() {	
    //Streammux properties
    g_object_set(G_OBJECT(streammux),
                 "width", config.muxer_width,
                 "height", config.muxer_height,
                 "batched-push-timeout", 40000,
                 "batch-size", 1,
                 "live-source", 0,
                  NULL);

    //GIE properties
    g_object_set(G_OBJECT(pgie),
				 "config-file-path",
				  config.model_config_path.c_str(),
				  NULL);    	
    
    //set properties of tracker
    g_object_set(G_OBJECT(tracker),
                "ll-lib-file", config.tracker_lib_path.c_str(),
                "ll-config-file", config.tracker_config_path.c_str(),
                "tracker-width", config.tracker_width,
                "tracker-height", config.tracker_height, NULL);
    
    // setting memory type to UNIFIED_MEMORY here 
    g_object_set(G_OBJECT(convertor), 
                "nvbuf-memory-type", 3, NULL);   

    if(!config.enable_display){
        fs::create_directories("../videos/output");
        g_object_set(G_OBJECT(filesink), "location", "../videos/output/blurred_vid.mp4");
    }

    GstPad * convertor_src_pad = gst_element_get_static_pad(convertor, "src");
    if (!convertor_src_pad){
        SPDLOG_ERROR("Unable to get sink pad\n");
        return false;
    }
    else{
        gst_pad_add_probe(convertor_src_pad, GST_PAD_PROBE_TYPE_BUFFER, 
                         blurFacesInFrameProbe, (gpointer)this, nullptr);        
        SPDLOG_INFO("added face blur probe to nvvideo convertor element");
    }
    return true;
}

bool FaceBlurPipeline::linkElements() {    
    if(config.enable_display){
        gst_bin_add_many(GST_BIN(pipeline), pgie, tracker, convertor, nvdsosd, videosink, nullptr);
        if (!gst_element_link_many(streammux, pgie, tracker, convertor, nvdsosd, videosink, NULL)) {
            SPDLOG_ERROR("Could not link one or more elements in case of streaming enabled");
            return false;
        }
        return true;
    }
    else{
        gst_bin_add_many(GST_BIN(pipeline), pgie, tracker, convertor, nvdsosd, convertor2, encoder, h264parser, muxer, filesink, nullptr);
        if (!gst_element_link_many(streammux, pgie, tracker, convertor, nvdsosd, convertor2, encoder, h264parser, muxer, filesink, NULL)) {
            SPDLOG_ERROR("Could not link one or more elements in case of streaming disabled");
            return false;
        }
        return true;
    }
}



bool FaceBlurPipeline::start() {
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