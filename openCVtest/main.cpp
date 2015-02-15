//
//  main.cpp
//  GestureRecognition
//
//  Created by LucyLin on 1/30/15.
//  Copyright (c) 2015 LucyLin. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>


using namespace std;
using namespace cv;

Mat src; //Mat src_gray; Mat src_bw;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
vector<pair<Point,double> > palm_centers;
pair<float, float> angleAndAveLen (const Point &v1, const Point &v2);
vector<int> codes;

/// Function header
void thresh_callback(int, void* );
void showImage(Mat img, string name);
double dist(Point x,Point y);
pair<Point,double> circleFromPoints(Point p1, Point p2, Point p3);
float angleBetween(const Point &p1, const Point &p2, const Point &center);
Mat imgPreprocess(Mat src, float newSize);
pair<vector<int>,vector<int>> inspect(Mat img);
int findPosition(const Point &p, float w, float y);


static void help()
{
    printf("\nDo background segmentation, especially demonstrating the use of cvUpdateBGStatModel().\n"
           "Learns the background at the start and then segments.\n"
           "Learning is togged by the space key. Will read from file or camera\n"
           "Usage: \n"
           "			./bgfg_segm [--camera]=<use camera, if this key is present>, [--file_name]=<path to movie file> \n\n");
}

const char* keys =
{
    "{c |camera   |true    | use camera or not}"
    "{fn|file_name|tree.avi | movie file             }"
};


int main(int argc, const char * argv[])
{
    ifstream file("/Users/lucylin/Desktop/openCVtest/path.txt");
    ifstream code("/Users/lucylin/Desktop/openCVtest/code.txt");
    
    
    vector<int> pw;
    vector<int> resetpw;
    vector<int> record; //valid input siries
    string str;
    bool loggedin = false;
    while (getline(file, str))
    {
        src = imread( str, 1 );

        //denoise original image
        Mat img = imgPreprocess(src, 0.25);
        
        pair<vector<int>,vector<int>> result = inspect(img);
        //too many objects
        if (result.first.size()>2)
        cout<<"detection result: "<<result.first[0]<<" "<<result.second[0]<<endl;
        
         //determine logic of the sequence
    }
   
    
    
//    help();
//    
//    VideoCapture cap;
//    bool update_bg_model = true;
//    
//    string file = "/Users/lucylin/Movies/test1.mov";
//    cap.open(file.c_str());
//    
//    if( !cap.isOpened() )
//    {
//        printf("can not open camera or video file\n");
//        return -1;
//    }
//    
//    namedWindow("image", CV_WINDOW_NORMAL);
//    namedWindow("foreground mask", CV_WINDOW_NORMAL);
//    namedWindow("foreground image", CV_WINDOW_NORMAL);
//    namedWindow("mean background image", CV_WINDOW_NORMAL);
//    
//    BackgroundSubtractorMOG2 bg_model;//(100, 3, 0.3, 5);
//    
//    Mat img, fgmask, fgimg;
//    
//    for(;;)
//    {
//        cap >> img;
//
//        
//        if( img.empty() ) {
//            cout<< "empty"<<endl;
//            break;
//        }
//        
//        //cvtColor(_img, img, COLOR_BGR2GRAY);
//        
//        if( fgimg.empty() )
//            fgimg.create(img.size(), img.type());
//        
//        //update the model
//        bg_model(img, fgmask, update_bg_model ? -1 : 0);
//        
//        fgimg = Scalar::all(0);
//        img.copyTo(fgimg, fgmask);
//
//        Mat bgimg;
//        bg_model.getBackgroundImage(bgimg);
//        
//        imshow("image", img);
//        imshow("foreground mask", fgmask);
//        imshow("foreground image", fgimg);
//        
//        if(!bgimg.empty())
//            imshow("mean background image", bgimg );
//        
//        char k = (char)waitKey(30);
//        if( k == 27 ) break;
//        if( k == ' ' )
//        {
//            update_bg_model = !update_bg_model;
//            if(update_bg_model)
//                printf("Background update is on\n");
//            else
//                printf("Background update is off\n");
//        }
//    }

    
    waitKey(0);
    return 0;

}


void showImage(Mat img, const string name) {
    namedWindow( name, CV_WINDOW_AUTOSIZE );
    //resize(img, img, Size(img.cols/4, img.rows/4));
    imshow( name, img );
    
}

double dist(Point x,Point y)
{
	return (x.x-y.x)*(x.x-y.x)+(x.y-y.y)*(x.y-y.y);
}

//This function returns the radius and the center of the circle given 3 points
//If a circle cannot be formed , it returns a zero radius circle centered at (0,0)
pair<Point,double> circleFromPoints(Point p1, Point p2, Point p3)
{
	double offset = pow(p2.x,2) +pow(p2.y,2);
	double bc =   ( pow(p1.x,2) + pow(p1.y,2) - offset )/2.0;
	double cd =   (offset - pow(p3.x, 2) - pow(p3.y, 2))/2.0;
	double det =  (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x)* (p1.y - p2.y);
	double TOL = 0.0000001;
	if (abs(det) < TOL) { cout<<"POINTS TOO CLOSE"<<endl;return make_pair(Point(0,0),0); }
    
	double idet = 1/det;
	double centerx =  (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet;
	double centery =  (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet;
	double radius = sqrt( pow(p2.x - centerx,2) + pow(p2.y-centery,2));
    
	return make_pair(Point(centerx,centery),radius);
}

float angleBetween(const Point &p1, const Point &p2, const Point &center)
{
    Point v1 = p1 - center;
    Point v2 = p2 - center;
    float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);
    
    float dot = v1.x * v2.x + v1.y * v2.y;
    
    float a = dot / (len1 * len2);
    
    if (a >= 1.0)
        return 0.0;
    else if (a <= -1.0)
        return 180;
    else
        return acos(a); // 0..PI
}

pair<float, float> angleAndAveLen (const Point &v1, const Point &v2)
{
    float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);
    
    float dot = v1.x * v2.x + v1.y * v2.y;
    
    float a = dot / (len1 * len2);
    
    if (a >= 1.0)
        return make_pair(0.0, (len1+len2)/2);
    else if (a <= -1.0)
        return make_pair(180, (len1+len2)/2);
    else
        return make_pair(acos(a), (len1+len2)/2); // 0..PI

}

Mat imgPreprocess(Mat src, float newSize)
{
    Mat src_gray; Mat src_bw;
    resize(src, src, Size(int(src.cols * newSize), int(src.rows * newSize)));
    cvtColor( src, src_gray, CV_BGR2GRAY );
    blur( src_gray, src_gray, Size(5,5) );
    threshold(src_gray, src_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    erode(src_bw, src_bw, Mat(), Point(-1, -1), 3);
    dilate(src_bw, src_bw, Mat());
    blur( src_bw, src_bw, Size(3,3) );
    
    //set 4 boundaries to 0 so that we can have closed contour
    copyMakeBorder( src_bw, src_bw, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0,0,0) );
    //    showImage(src, "Source")
    //showImage(src_bw, "bw");
    return src_bw;
    
}

int findPosition(const Point &p, float w, float h)
{
    int x, y;
    if (p.x < w/3)
        x = 0;
    else if (p.x>w/3*2)
        x = 2;
    else
        x = 1;
    
    if (p.y < h/3)
        y = 0;
    else if (p.y > h/3*2)
        y = 2;
    else
        y = 1;

    return 3*y+x;

}

pair<vector<int>, vector<int>> inspect(Mat src_bw)
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    pair<vector<int>, vector<int>> result;
    
    /// Detect edges using canny
    Canny( src_bw, canny_output, thresh, thresh*2);
    
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );
    //showImage(canny_output, "canny");
    
    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );
    }
    
    // Find the convex hull object and defects for "each" contour
    vector<vector<Point> >hullsP( contours.size() );
    vector<vector<Vec4i> >defects( contours.size() );
    vector<vector<int> > hullsI(contours.size());
    for( int i = 0; i < contours.size(); i++ )
    {
        //ignore noisy contour
        if (contours[i].size() < 500)
            continue;
        convexHull(Mat(contours[i]), hullsI[i], false, false);
        
        //////////////////////////////////////////////////////
        //delete noisy convex defects, try to find palm center
        //////////////////////////////////////////////////////
        convexHull(Mat(contours[i]), hullsP[i], false, true);
        convexityDefects(contours[i], hullsI[i], defects[i]);
        if(defects[i].size()>=3)
        {
            //cout<<"#defects: "<<defects[i].size()<<endl;
            vector<Vec4i>::iterator d =defects[i].begin();
            vector<Point> palm_points;
            Point rough_palm_center;
            while( d!=defects[i].end() )
            {
                Vec4i& v=(*d);
                if(abs(v[1]-v[0]) < 5 || v[3]/256 < 5) {
                    defects[i].erase(d);
                }
                else {
                    int startidx=v[0];; Point ptStart( contours[i][startidx] );
                    int endidx=v[1]; Point ptEnd( contours[i][endidx] );
                    int faridx=v[2]; Point ptFar( contours[i][faridx] );
                    //Sum up all the hull and defect points to compute average
                    rough_palm_center+=ptFar+ptStart+ptEnd;
                    palm_points.push_back(ptFar);
                    //                palm_points.push_back(ptStart);
                    //                palm_points.push_back(ptEnd);
                    d++;
                }
            }
            
            //Get palm center by 1st getting the average of all defect points, this is the rough palm center,
            //Then U chose the closest 3 points ang get the circle radius and center formed from them which is the palm center.
            rough_palm_center.x/=defects[i].size()*3;
            rough_palm_center.y/=defects[i].size()*3;
            Point closest_pt=palm_points[0];
            vector<pair<double,int> > distvec;
            for(int i=0;i<palm_points.size();i++)
                distvec.push_back(make_pair(dist(rough_palm_center,palm_points[i]),i));
            sort(distvec.begin(),distvec.end());
            
            //Keep choosing 3 points till you find a circle with a valid radius
            //As there is a high chance that the closes points might be in a linear line or too close that it forms a very large circle
            pair<Point,double> soln_circle;
            for(int i=0;i+2<distvec.size();i++)
            {
                Point p1=palm_points[distvec[i+0].second];
                Point p2=palm_points[distvec[i+1].second];
                Point p3=palm_points[distvec[i+2].second];
                soln_circle=circleFromPoints(p1,p2,p3);//Final palm center,radius
                if(soln_circle.second!=0)
                    break;
            }
            
            //Find avg palm centers for the last few frames to stabilize its centers, also find the avg radius
            palm_centers.push_back(soln_circle);
            if(palm_centers.size()>10)
                palm_centers.erase(palm_centers.begin());
            
            Point palm_center;
            double radius=0;
            for(int i=0;i<palm_centers.size();i++)
            {
                palm_center+=palm_centers[i].first;
                radius+=palm_centers[i].second;
            }
            palm_center.x/=palm_centers.size();
            palm_center.y/=palm_centers.size();
            radius/=palm_centers.size();
            
            //Draw the palm center and the palm circle
            //The size of the palm gives the depth of the hand
            circle(drawing,palm_center,5,Scalar(144,144,255),3);
            circle(drawing,palm_center,radius,Scalar(144,144,255),2);
            circle(drawing, rough_palm_center, 5, Scalar(144,144,255),3);
            cout<<"center"<<palm_center.x<<endl;
            
            //Detect fingers by finding points that form an almost isosceles triangle with certain thesholds
            int no_of_fingers=0;
            d =defects[i].begin();
            while( d!=defects[i].end() )
            {
                Vec4i& v=(*d);
                int startidx=v[0]; Point ptStart( contours[i][startidx] );
                int endidx=v[1]; Point ptEnd( contours[i][endidx] );
                int faridx=v[2]; Point ptFar( contours[i][faridx] );
                //X o--------------------------o Y
                //                double Xdist=sqrt(dist(palm_center,ptFar));
                //                double Ydist=sqrt(dist(palm_center,ptStart));
                //                double length=sqrt(dist(ptFar,ptStart));
                //                double retLength=sqrt(dist(ptEnd,ptFar));
                Scalar colors = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                
                //check angle between 2 possible fingers, rule out the ones that have angle > 45 (assume angle between 2 fingers < 45)
                float angle = angleBetween(ptStart, ptEnd, ptFar);
                //cout<<"angle = "<<angle<<endl;
                if(angle < 1) {
                    no_of_fingers++;
                    line( drawing, ptEnd, ptStart, colors, 3);
                    line( drawing, ptStart, ptFar, CV_RGB(0,255,0), 1 );
                    circle( drawing, ptFar,   4, Scalar(0,255,100), 2 );
                }
                
                //                //drawContours( drawing, hullsP, i, colors, 1, 8, vector<Vec4i>(), 0, Point() );
                //                line( drawing, ptEnd, ptStart, colors, 3);
                //                line( drawing, ptStart, ptFar, CV_RGB(0,255,0), 1 );
                //                circle( drawing, ptFar,   4, Scalar(0,255,100), 2 );
                //
                //                //Play with these thresholds to improve performance
                //                //if(length<=3*radius&&Ydist>=0.4*radius&&length>=10&&retLength>=10&&max(length,retLength)/min(length,retLength)>=0.8)
                //                //if (length<radius*2)
                //                {
                //
                //                    if(min(Xdist,Ydist)/max(Xdist,Ydist)<=0.8)
                //                    {
                //                        if((Xdist>=0.1*radius&&Xdist<=1.3*radius&&Xdist<Ydist)||(Ydist>=0.1*radius&&Ydist<=1.3*radius&&Xdist>Ydist))
                //                        {
                //                            //line( drawing, ptEnd, ptFar, Scalar(0,255,0), 1 );
                //                            no_of_fingers++;
                //                            circle( drawing, ptFar,   6, Scalar(255,0,100), 2 );
                //                        }
                //                    }
                //                }
                
                d++;
            }
            
            int pos = findPosition(palm_center, src_bw.cols, src_bw.rows);
            
            //no_of_fingers=min(5,no_of_fingers);
            if (no_of_fingers >=1) {
                no_of_fingers++;
                cout<<"NO OF FINGERS: "<<no_of_fingers<<endl;
                result.first.push_back(no_of_fingers);
                result.second.push_back(pos);
            }
            //may be fist, one finger or nothing
            else {
                cout<<"may be fist, one finger or nothing"<<endl;
                float minDist = 99999;
                Vector<Vec4i> singleFinger;
                pair<Vec4i, Vec4i> closestPair;
                //                int closestPair;
                Vec4i lastv = defects[i][defects[i].size()-1];
                for(int j = 0; j<defects[i].size(); j++)
                {
                    int faridx_1=lastv[2]; Point ptFar1( contours[i][faridx_1] );
                    Vec4i v2 = defects[i][j];
                    int faridx_2=v2[2]; Point ptFar2( contours[i][faridx_2] );
                    float dis = dist(ptFar1, ptFar2);
                    if(dis < minDist) {
                        minDist = dis;
                        closestPair = make_pair(lastv, v2);
                    }
                    lastv = v2;
                }
                
                //check angle between 2 possible fingers, they should be almost parallel
                Point p11( contours[i][closestPair.first[0]]);
                Point p12( contours[i][closestPair.first[1]]);
                Point p13( contours[i][closestPair.first[2]]);
                circle( drawing, p13,   6, Scalar(255,255,100), 2 );
                circle( drawing, p11,   6, Scalar(255,255,100), 2 );
                
                Point p21( contours[i][closestPair.second[0]]);
                Point p22( contours[i][closestPair.second[1]]);
                Point p23( contours[i][closestPair.second[2]]);
                circle( drawing, p23,   6, Scalar(255,255,100), 2 );
                circle( drawing, p21,   6, Scalar(255,255,100), 2 );
                circle( drawing, Point (10, 20),   6, Scalar(255,255,255), 2 );
                
                pair< float, float > pair1 = angleAndAveLen(p12-p13, p21-p23);
                pair< float, float > pair2 = angleAndAveLen(p11-p13, p22-p23);
                //cout << "radius = "<<radius <<endl;
                //cout << pair1.first<<" "<<pair1.second<<endl;
                //cout << pair2.first<<" "<<pair2.second<<endl;
                
                //also, the finger length has to fall between 0.5~1.5 palm radius
                
                
                if((pair1.first < 0.5 && pair1.second < radius * 1.5 && pair1.second > radius * 0.5)
                   || (pair2.first < 0.5 && pair2.second < radius * 1.5 && pair2.second > radius * 0.5)) {
                    cout<<"#finger = 1";
                    result.first.push_back(1);
                    result.second.push_back(pos);

                    //reassess palm center ?
                    
                }
                else {
                    cout<<"fist or others";
                    result.first.push_back(0);
                    result.second.push_back(pos);
                }
                
            }
            
            
        }
        
    }
    
    showImage(drawing, "convex");
    
    // Draw contours + hull results
    RNG rng;
    Mat hullDrawing = Mat::zeros( drawing.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( hullDrawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        //drawContours( hullDrawing, hullsP, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        
        size_t count = contours[i].size();
        std::cout<<"contour size : "<<count<<std::endl;
        if( count < 300 )
            continue;
        vector<Vec4i>::iterator d =defects[i].begin();
        
        while( d!=defects[i].end() )
        {
            Vec4i& v=(*d);
            int startidx=v[0];
            Point ptStart( contours[i][startidx] ); // point of the contour where the defect begins
            int endidx=v[1];
            Point ptEnd( contours[i][endidx] ); // point of the contour where the defect ends
            int faridx=v[2];
            Point ptFar( contours[i][faridx] );// the farthest from the convex hull point within the defect
            int depth = v[3] / 256; // distance between the farthest point and the convex hull
            
            //if(depth > 30 && depth < 300 )
            //{
            //cout<<"depth = "<<depth<<endl;
            
            line( hullDrawing, ptStart, ptFar, CV_RGB(0,255,0), 2 );
            line( hullDrawing, ptEnd, ptFar, CV_RGB(0,255,0), 1 );
            //line( hullDrawing, ptStart, ptEnd, CV_RGB(0,255,0), 2 );
            circle( hullDrawing, ptStart,   4, Scalar(255,0,100), 1 );
            circle( hullDrawing, ptEnd,   4, Scalar(255,0,100), 1 );
            circle( hullDrawing, ptFar,   4, Scalar(100,0,255), 1 );
            //}
            d++;
        }
        
    }
    
    
    
    // Show in a window
    showImage(hullDrawing, "hull");

    return result;
}
