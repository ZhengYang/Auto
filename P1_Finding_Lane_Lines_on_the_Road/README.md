# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. First, I converted the images to HSL color space and filter out the yellow and white color. Second, I converted the filtered images to grayscale. The third step is applying ganssian blur on the image, and the fourth step is detecting edges using canny edge. Step No. 5 is using Hough transform to extract major lines from the edge image. Lastly, I group the lines into 2 groups, left leaning and right leaning lines. The average of each group represents the detected lane.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function in the following ways. First, the lines are categorized into two groups based on the signs of their gridient. 

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
