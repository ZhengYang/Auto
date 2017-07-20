# **Finding Lane Lines on the Road** 


The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. First, I converted the images to HSL color space and filter out the yellow and white color. Second, I converted the filtered images to grayscale. The third step is applying ganssian blur on the image, and the fourth step is detecting edges using canny edge. Step No. 5 is using Hough transform to extract major lines from the edge image. Lastly, I group the lines into 2 groups, left leaning and right leaning lines. The average of each group represents the detected lane.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function in the following ways.
First, the lines are categorized into two groups based on the signs of their gridient. 
Lines having positive (*+*) gradient are put into a group from which we will compute the _average line_ to represent *right* lane.
At the same time, lines having negative (*-*) gradient are put into a group from which we will compute the _average line_ to represent *left* lane.
To compute the _average line_, I need 2 piece of information: the average gradient and the average interception to y axis. This is because a line can be uniquely identified by `y = kx + b` where `k` is the gradient and `b` is the y-intercept. These two quantities can be easily computed and the resulting line will be the _average line_.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when there are markings on the road.
For instance, in some part of the road, there will be markings saying 'SLOWDOWN'.
This may potentially lead the pipeline to identify irrelavent lines, and hence affecting the accuracy of lane detection.
Similarly, box junctions will also affect the result. Both examples can be found in the photo below.

![Fig 1][http://www.todayonline.com/sites/default/files/styles/photo_gallery_image_lightbox/public/16959708.JPG]

Another shortcoming could be when driving at night and the road is wet, the reflections may create a lot of noise which makes lane not very visible. See the picture below.

![Fig 2][https://honestuniverse.files.wordpress.com/2015/06/wet-road-reflections1.jpg]


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
