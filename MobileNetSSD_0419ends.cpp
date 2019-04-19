#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "net.h"

#include "../model/tumble.id.h"
#include "../model/tumble.mem.h"
#include <iostream>
using namespace std;

struct Object//object box
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    float action;//confidence
    int action_index;//class_index
};

static int detect_mobilenet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net mobilenet;
    mobilenet.load_param("../../model/MobileNetSSD_deploy.param");
    mobilenet.load_model("../../model/MobileNetSSD_deploy.bin");
    const int target_size = 300;
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    //printf("image_ori:%d %d %d\n", bgr.rows, bgr.cols, bgr.type());//height,width
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals,norm_vals);
    ncnn::Extractor ex = mobilenet.create_extractor();
    ex.set_num_threads(4);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("detection_out",out); 
    objects.clear();
    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);
        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;
        //printf("crop:%d %d %d %d\n", values[2] * img_w, values[3] * img_h, values[4] * img_w,values[5] * img_h);//
        //printf("crop_size:%d %d\n", object.rect.width, object.rect.height);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f \n", object.label, object.prob,object.rect.x, object.rect.y, object.rect.width, object.rect.height );

        cv::Mat bgr_gray;//gray
        cv::cvtColor(bgr,bgr_gray,CV_BGR2GRAY);
        //printf("image:%d %d %d\n", bgr_gray.rows, bgr_gray.cols, bgr_gray.type());//
        int int_x=((int)object.rect.x>1)?(int)object.rect.x:1;
        int int_y=((int)object.rect.y>1)?(int)object.rect.y:1;
        int int_x2=((int)(values[4] * img_w)>bgr.cols)?bgr.cols:(int)(values[4] * img_w);//bgr.cols
        int int_y2=((int)(values[5] * img_h)>bgr.rows)?bgr.rows:(int)(values[5] * img_h);//bgr.rows
        //int int_w = (int)(int_x2-int_x);
        //(int) int_w=int_x2-int_x;
        //int int_h = (int)(int_y2-int_y);
        //(int) int_h=int_y2-int_y;
        //printf("image_position x_y:%d %d %d %d\n",int_x,int_x2,int_y,int_y2);
        cv::Mat image_ROI= bgr_gray(cv::Range(int_y,int_y2),cv::Range(int_x,int_x2));
        //cv::imwrite("rotated_im.png", image_ROI);
        /*
        int nsize=(int)object.rect.width*(int)object.rect.height;
        unsigned char *buf =new unsigned char[nsize];
        unsigned char *pdst=buf;
        //printf("bgr.data:%d %d\n", bgr.data)
        for (int i=(int)object.rect.x;i<(int)(values[4] * img_w)+1;++i)//100
        {
            for (int j=(int)object.rect.y;j<(int)(values[5] * img_h)+1;j++)//50
            {
                *pdst++ = bgr_gray.data[i*bgr_gray.cols+j];
                //printf("%d,%d:%hhd\t",i,j,bgr.data[i*bgr.cols+j]);
            }
        }
        */
        //printf("roi:%d %d %d\n",image_ROI.cols, image_ROI.rows,image_ROI.type());//
        ncnn::Mat box = ncnn::Mat::from_pixels_resize(image_ROI.data,ncnn::Mat::PIXEL_GRAY,  image_ROI.cols, image_ROI.rows, 56, 56);
        //printf("box:%d %d %d\n", box.w, box.h, box.c);//
        /*
        //for (int i=0;i<nsize;i++)
        //    printf("%d:%hhd\t",i,buf[i]);
        printf("------------\n");
        //printf("%d::%d",object.rect.width,object.rect.height);
        ncnn::Mat box = ncnn::Mat::from_pixels_resize(buf, ncnn::Mat::PIXEL_BGR2GRAY, object.rect.width, object.rect.height, 56, 56);
        */
        const float mean_vals[1]={127.5f};
        const float norm_vals[1]={1/255.f};
        box.substract_mean_normalize(mean_vals,norm_vals);
        //delete []buf;
        ncnn::Net tumble_net;
        tumble_net.clear();
        tumble_net.load_param(tumble_param_bin);
        tumble_net.load_model(tumble_bin);
        ncnn::Mat out;
        ncnn::Extractor box_net = tumble_net.create_extractor();
        box_net.set_num_threads(4);
        box_net.set_light_mode(true);
        box_net.input(tumble_param_id::BLOB_data,box);
        box_net.extract(tumble_param_id::BLOB_softmax,out);
        float* pFv = NULL;
        pFv=out.row(0);
        int maxi_c=0;
        float maxf_c=pFv[0];
        for (int c=1;c<out.w;c++)
        {
            if (pFv[c]>maxf_c)
            {
                maxf_c = pFv[c];
                maxi_c = c;
            }
        }
        object.action_index=maxi_c;//index
        object.action=maxf_c;//confidence
        objects.push_back(object);
    }



    return 0;
}

//CV_EXPORTS_W VideoCapture::VideoCapture()
static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};
    cv::Mat image = bgr.clone();
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
		//detect result
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f is tumble: %.2f %d\n", obj.label, obj.prob,obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height , obj.action , obj.action_index );
        //cv::Mat imageROI;
        //imageROI=image(cv::Rect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height));
        //cv::imwrite("rotated_im_%d.png", i , imageROI);
        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));
        //cv::imwrite("../output/%d.png",i,imageROI);
        //cv::imwrite("../output/%d.png", i , imageROI);
        //if (i==1):
        //	sr="a";
        //else:
        //	sr="b";
        //str=std::to_string(i);
        //string str=to_string(i);
        //imagename="../output/object_%d.png" % (i);
        //char image_name[50];
        //image_name = "../output/%d.jpg" % (i);
        //cv2.imwrite(image_name, imageROI)
        //imagename="../output/object_%d_%d.png" % (obj.label,i);
        //imagename << "../output/object_" << i << ".png"
        //cv::imwrite(imagename,imageROI);
        //cv::imwrite("../output/im_%d.png", i , imageROI);
        //cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));
        //cv::imwrite("../output/rotated_im_%d.png", i , imageROI);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;


        char text2[256];
        sprintf(text2, "tumble %.1f%%", obj.prob * 100);
        //char text3[256];
        //sprintf(text3, "fight %.1f%%", obj.prob * 100);   
        if ( obj.rect.width <= obj.rect.height && ( obj.label ==4 || obj.label ==3 || obj.label ==1  || obj.label ==12 || obj.label ==15 ))
        {
			cv::rectangle(image, cv::Rect(cv::Point(x, y),cv::Size(label_size.width, label_size.height + baseLine)),cv::Scalar(255, 255, 255), CV_FILLED);
			//fprintf(stderr, "%.2f %.2f: %.2f x %.2f is person: %.1f%% is tumble: 0.00 \n", obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height ,obj.prob*100 );	
        }
		else if ( obj.rect.width > obj.rect.height && ( obj.label ==4 || obj.label ==3 || obj.label ==1  || obj.label ==12 || obj.label ==15 ))
		{
			cv::rectangle(image, cv::Rect(cv::Point(x, y),cv::Size(label_size.width, label_size.height + baseLine)),cv::Scalar(255, 255, 255), CV_FILLED);
			cv::putText(image, text2, cv::Point(x, y + label_size.height),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			//fprintf(stderr, "%.2f %.2f: %.2f x %.2f is person: %.1f%% is tumble: %d confidence: %.2f\n", obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height ,obj.prob*100, obj.action_index, obj.action );	
			fprintf(stderr, "%.2f %.2f: %.2f x %.2f is tumble: %.1f%% \n", obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height ,obj.prob*100 );	
		}
		else
		{
			cout<<"No person in this picture "<<endl;
		}

        /*
        //fight_part
        else if (obj.label ==15 && ( obj.rect.width<obj.rect.height && 2*obj.rect.width>obj.rect.height ))
        {    cv::rectangle(image, cv::Rect(cv::Point(x, y),cv::Size(label_size.width, label_size.height + baseLine)),cv::Scalar(255, 255, 255), CV_FILLED);
             cv::putText(image, text3, cv::Point(x, y + label_size.height),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
        */

        /*
        //ori output
        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);
        cv::putText(image, text2, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        */
    }
    //cv::imshow("image", image);
    //cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
    const char* imagepath = argv[1];
    cv::VideoCapture capture(imagepath);//
    if (!capture.isOpened())
        cout<<"cannot open"<<endl;
    //double rate=capture.get(CV_CAP_PROP_FPS);
    int rate=capture.get(CV_CAP_PROP_FPS);
    cout<<"rate:"<<rate<<endl;
    bool stop(false);
    int delay= 1000/rate;
    //double msec=capture.get(CV_CAP_PROP_POS_MSEC);
    int frame_cnt= 0;

    while (!stop)
    {
        cv::Mat m;
        capture >> m;
        if (m.empty())
        {   
            break;//fprintf(stderr, "cv::imread %s failed\n", imagepath);
            //return -1;
        }
        //cv::waitKey(40);
        //cout << "" << endl;
        if (frame_cnt % rate == 0)
	//if(true)
	{	
		cout<<" "<<endl;
        	cout << "Video Time:"<<floor(capture.get(CV_CAP_PROP_POS_MSEC)/1000)  << " s" <<endl;
	        std::vector<Object> objects;
	        detect_mobilenet(m, objects);
	        draw_objects(m, objects);
	}
	++frame_cnt;
	//cout<<"frame"<<frame_cnt <<endl;
        if (cv::waitKey(delay)>=0)
            stop = true;
    }
    capture.release();
    return 0;
}
