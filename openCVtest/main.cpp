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
#include <iostream>
#include <stdio.h>
#include <stdlib.h>


using namespace std;
using namespace cv;

Mat src; Mat src_gray; Mat src_bw;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
vector<pair<Point,double> > palm_centers;

/// Function header
void thresh_callback(int, void* );
void showImage(Mat img, string name);
double dist(Point x,Point y);
pair<Point,double> circleFromPoints(Point p1, Point p2, Point p3);
float angleBetween(const Point &p1, const Point &p2, const Point &center);

int main(int argc, const char * argv[])
{
    
    src = imread( "/Users/lucylin/Dropbox/class/VI/img/med_4.jpg", 1 );
    resize(src, src, Size(src.cols/4, src.rows/4));
    cvtColor( src, src_gray, CV_BGR2GRAY );
    blur( src_gray, src_gray, Size(5,5) );
    threshold(src_gray, src_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    erode(src_bw, src_bw, Mat(), Point(-1, -1), 3);
    dilate(src_bw, src_bw, Mat());
    blur( src_bw, src_bw, Size(3,3) );
    //set 4 boundaries to 0 so that we can have closed contour
    copyMakeBorder( src_bw, src_bw, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0,0,0) );
    
    namedWindow( "Source", CV_WINDOW_AUTOSIZE );
    imshow( "Source", src );
    
    showImage(src_bw, "bw");
    
    createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
    thresh_callback( 0, 0 );
    

    
    waitKey(0);
    return 0;
}

void thresh_callback(int, void* )
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
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
    
    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    
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
        //if a segment is too short, remove it.
//        vector<int>::iterator iter =hullsI[i].begin();
//        int last = hullsI[i][0];
//        iter++;
//        while( iter!=hullsI[i].end() )
//        {
//            if(last-*iter <  5)
//            {
//                hullsI[i].erase(iter);
//            } else {
//                last = *iter;
//                iter++;
//            }
//            cout<<hullsI[i].size()<<endl;
//        }
        
        //////////////////////////////////////////////////////
        //delete noisy convex defects, try to find palm center
        //////////////////////////////////////////////////////
        convexHull(Mat(contours[i]), hullsP[i], false, true);
        convexityDefects(contours[i], hullsI[i], defects[i]);
        if(defects[i].size()>=3)
        {
            cout<<"#defects: "<<defects[i].size()<<endl;
            vector<Vec4i>::iterator d =defects[i].begin();
            vector<Point> palm_points;
            Point rough_palm_center;
            while( d!=defects[i].end() )
            {
                Vec4i& v=(*d);
                if(abs(v[1]-v[0]) < 5) {
                    defects[i].erase(d);
                }
                else {
                int startidx=v[0];; Point ptStart( contours[i][startidx] );
                int endidx=v[1]; Point ptEnd( contours[i][endidx] );
                int faridx=v[2]; Point ptFar( contours[i][faridx] );
                //Sum up all the hull and defect points to compute average
                rough_palm_center+=ptFar+ptStart+ptEnd;
                palm_points.push_back(ptFar);
                //palm_points.push_back(ptStart);
                //palm_points.push_back(ptEnd);
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
            
            //Detect fingers by finding points that form an almost isosceles triangle with certain thesholds
            int no_of_fingers=0;
            d =defects[i].begin();
            while( d!=defects[i].end() )
            {
                Vec4i& v=(*d);
                int startidx=v[0];; Point ptStart( contours[i][startidx] );
                int endidx=v[1]; Point ptEnd( contours[i][endidx] );
                int faridx=v[2]; Point ptFar( contours[i][faridx] );
                //X o--------------------------o Y
//                double Xdist=sqrt(dist(palm_center,ptFar));
//                double Ydist=sqrt(dist(palm_center,ptStart));
//                double length=sqrt(dist(ptFar,ptStart));
//                double retLength=sqrt(dist(ptEnd,ptFar));
//                Scalar colors = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                
                //check angle between 2 possible fingers, rule out the ones that have angle > 45 (assume angle between 2 fingers < 45)
                float angle = angleBetween(ptStart, ptEnd, ptFar);
                cout<<"angle = "<<angle<<endl;
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
            
            //no_of_fingers=min(5,no_of_fingers);
            if (no_of_fingers >=1)
                no_of_fingers++;
            cout<<"NO OF FINGERS: "<<no_of_fingers<<endl;
            
            
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
        std::cout<<"Count : "<<count<<std::endl;
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
                cout<<depth<<endl;

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
