#include "stdafx.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
static int fps_num=6;
static int fps=1;
#define  INTENSITY_T 5
//#define  HT_P
#define  MAX_D 255
#define	 MIN_D 0
#define PI 3.141592653
#define C 4
Mat image,disparity,v_disparity,v_disparity_result,img_result,dsp_result;
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))
int bound(short i,short a,short b)
{
	return min(max(i,min(a,b)),max(a,b));
}
void on_mouse(int event, int x, int y, int flags, void* param)
{
	Mat disparity_t,image_t,v_disparity_t,img_result_t,dsp_result_t;	
	image.copyTo(image_t);	
	img_result.copyTo(img_result_t);
	dsp_result.copyTo(dsp_result_t);
	cvtColor(disparity,disparity_t,CV_GRAY2RGB);
	//cvtColor(v_disparity,v_disparity_t,CV_GRAY2RGB);
	v_disparity_result.copyTo(v_disparity_t);
	CvFont font;
	CvSize text_size;
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX,0.4,0.4,0,1,CV_AA);
	x=bound(x,0,disparity.cols-1);
	y=bound(y,0,disparity.rows-1);
	Point pt(x,y);
	Point dpt_v(disparity.at<uchar>(pt),y);
	char temp[30];
	sprintf(temp,"%d",disparity.at<uchar>(pt));
	int baseline;
	cvGetTextSize(temp,&font,&text_size,&baseline);
	Point pt_t(bound(pt.x,0,disparity.cols-text_size.width),bound(pt.y,text_size.height+baseline,disparity.rows-1-baseline));
	if (event==CV_EVENT_MOUSEMOVE)
	{
		circle(disparity_t,pt,2,Scalar(0,0,255),CV_FILLED,CV_AA,0);
		circle(image_t,pt,2,Scalar(0,0,255),CV_FILLED,CV_AA,0);
		putText(img_result_t,temp,pt_t,font.font_face,font.hscale,Scalar(0,0,255));
		circle(img_result_t,pt,2,Scalar(0,0,255),CV_FILLED,CV_AA,0);
		circle(dsp_result_t,pt,2,Scalar(0,0,255),CV_FILLED,CV_AA,0);
		circle(v_disparity_t,dpt_v,2,Scalar(0,0,255),CV_FILLED,CV_AA,0);
		imshow("disparity",disparity_t);
		imshow("image",image_t);
		imshow("result",img_result_t);
		imshow("disparity_result",dsp_result_t);
		imshow("v-disparity",v_disparity_t);
	}
	else if(event ==CV_EVENT_LBUTTONDOWN)
	{
		
			
			circle(disparity,pt,2,Scalar(0),CV_FILLED,CV_AA,0);
			circle(image,pt,2,Scalar(0,255,0),CV_FILLED,CV_AA,0);	
			circle(img_result,pt,2,Scalar(0,255,0),CV_FILLED,CV_AA,0);
			circle(dsp_result_t,pt,2,Scalar(0,0,255),CV_FILLED,CV_AA,0);
			//circle(v_disparity,dpt_v,2,Scalar(255),CV_FILLED,CV_AA,0);
			circle(v_disparity_result,dpt_v,2,Scalar(0,255,0),CV_FILLED,CV_AA,0);
	}
}
int _tmain(int argc, _TCHAR* argv[])
{	
	//main loop
	bool flag=true;
	while(flag)
	{
		cout<<"当前第 "<<fps<<"张图像"<<endl;
		if (fps>fps_num)
		{
			flag=false;
			continue;;
		}
		//1.read images
		string filename=format(".\\sample\\%d\\left.png",fps);
		image=imread(filename);
		filename=format(".\\sample\\%d\\disparity_8uc1.png",fps);
		disparity=imread(filename);
		assert(image.channels()==3);		
		if (disparity.channels()!=1)
		{
			cvtColor(disparity,disparity,CV_RGB2GRAY);
		}
		assert(disparity.channels()==1);
		if (!image.data||!disparity.data)
		{
			printf("没有图片\n");
			fps++;
			continue;
		}
		//2.calculate v-disparity
		v_disparity=Mat::zeros(disparity.rows,256,CV_8UC1);
		Mat v_disparity_data=Mat::zeros(disparity.rows,256,CV_16UC1);
		for(int i=0; i < disparity.rows; i++)
		{
			uchar* data_s=disparity.ptr<uchar>(i);
			ushort* data_d=v_disparity_data.ptr<ushort>(i);
			for(int j=0; j < disparity.cols; j++)
			{
				// increase v_disparity counter
				if(data_s[j]<=0||data_s[j]>=255) continue;
				data_d[data_s[j]] += 1;

			}
		}
		normalize(v_disparity_data, v_disparity, 0, 255, NORM_MINMAX, CV_8UC1);
		cvtColor(v_disparity,v_disparity_result,CV_GRAY2RGB);
		//3.filter v-disparity
		Mat v_disparity_filter=Mat::zeros(v_disparity.rows,v_disparity.cols,CV_8UC1);
		vector<Point> points;
		for(int i=0;i<v_disparity.cols;i++)
		{
			int max_value=0;
			int max_index=0;			
			for(int j=0;j<v_disparity.rows;j++)
			{
				int value=v_disparity.at<uchar>(j,i);
				if(value>max_value)
				{
					max_value=value;
					max_index=j;
				}				
			}
			if(max_value>INTENSITY_T){
				v_disparity_filter.at<uchar>(max_index,i)=v_disparity.at<uchar>(max_index,i);
				v_disparity_result.at<Vec3b>(max_index,i)=Vec3b(0,255,0);
				points.push_back(Point(i,max_index));
			}			
		}		
		//4.line fitting
		double hough_rho = 1;
		double hough_theta = CV_PI/180;
		int hough_threshold = 10;//10
#ifdef HT_P
		vector<Vec4i> lines;		
		double hough_min_line_lenght = 10;
		double hough_max_line_gap = 100;//20
		HoughLinesP(v_disparity_filter, lines, hough_rho, hough_theta, hough_threshold, hough_min_line_lenght, hough_max_line_gap);
		for( size_t i = 0; i < lines.size(); i++ )
		{
			circle(v_disparity_result,Point(lines[i][0], lines[i][1]),2,Scalar(0,255,255),CV_FILLED,CV_AA,0);
			circle(v_disparity_result,Point(lines[i][2], lines[i][3]),2,Scalar(0,255,255),CV_FILLED,CV_AA,0);
			line( v_disparity_result, Point(lines[i][0], lines[i][1]),
				Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 1, 8 );
		}
#else
		Vec4f lines;		
		fitLine(points,lines,CV_DIST_WELSCH,0,0.01,0.01);
		double cos_theta = lines[0];
		double sin_theta = lines[1];
		double x0 = lines[2], y0 = lines[3];
	
		double k = sin_theta / cos_theta;
		double b = y0 - k * x0;
		double x1 = 0;
		double y1 = k * x1 + b;
		double y2 = v_disparity_result.rows-1;
		double x2 = (y2-b)/k;
		line(v_disparity_result, Point(x1,y1), Point(x2,y2), Scalar(0,0,255), 1, 8);		
		
		vector<double> ground_line;
		for(int i=0;i<disparity.rows;i++)
		{
			double d_exp=(i-b)/k;
			ground_line.push_back(d_exp);
		}
#endif
		//5.predict ground plane
		image.copyTo(img_result);		
		cvtColor(disparity,dsp_result,CV_GRAY2RGB);
		for(int i=0;i<img_result.rows;i++)
		{
			Vec3b *data_i=img_result.ptr<Vec3b>(i);
			uchar* data_d=disparity.ptr<uchar>(i);
			Vec3b *data_dr=dsp_result.ptr<Vec3b>(i);
			for(int j=0;j<img_result.cols;j++)
			{
				if(data_d[j]==0)continue;
				else
				{
					double r=ground_line[i]-data_d[j];
					double confidence=C*exp(-r*r/(2*C*C));
					if (confidence>0.5)
					{
						data_i[j]+=Vec3b(0,125,0);
						data_dr[j]+=Vec3b(0,125,0);
					}
				 }

			}
		}
		//imshow("image",image);
		//imshow("disparity",disparity);
		//imshow("v-disparity",v_disparity_result);
		imshow("result",img_result);
		//imshow("disparity_result",dsp_result);
		//imwrite(format("D:\\360安全浏览器下载\\sample\\%d\\v_disparity_result.png",fps),v_disparity_result);
		//imwrite(format("D:\\360安全浏览器下载\\sample\\%d\\left_result.png",fps),img_result);
		//imwrite(format("D:\\360安全浏览器下载\\sample\\%d\\disparity_8uc1_result.png",fps),dsp_result);
		//setMouseCallback("result",on_mouse,NULL);
		char key=waitKey(10);
		switch(key)
		{			
		case 'q':flag=false;
			break;					
		}	
		fps++;
	}
	return 0;
}

