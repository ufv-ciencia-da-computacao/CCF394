We have defined a function that classifies the pixels of an image as skin
or non-skin simply based on an interval of values (the minimum and
maximum hue, and the minimum and maximum saturation):
void detectHScolor(const cv::Mat& image, // input image
double minHue, double maxHue, // Hue interval
double minSat, double maxSat, // saturation
interval
cv::Mat& mask) { // output mask
// convert into HSV space
cv::Mat hsv;
cv::cvtColor(image, hsv, CV_BGR2HSV);
// split the 3 channels into 3 images
std::vector<cv::Mat> channels;
cv::split(hsv, channels);
// channels[0] is the Hue
// channels[1] is the Saturation
// channels[2] is the Value// Hue masking
cv::Mat mask1; // below maxHue
cv::threshold(channels[0], mask1, maxHue, 255,
cv::THRESH_BINARY_INV);
cv::Mat mask2; // over minHue
cv::threshold(channels[0], mask2, minHue, 255,
cv::THRESH_BINARY);
cv::Mat hueMask; // hue mask
if (minHue < maxHue)
hueMask = mask1 & mask2;
else // if interval crosses the zero-degree axis
hueMask = mask1 | mask2;
// Saturation masking
// between minSat and maxSat
cv::Mat satMask; // saturation mask
cv::inRange(channels[1], minSat, maxSat, satMask);
// combined mask
mask = hueMask & satMask;
}
Having a large set of skin (and non-skin) samples at our disposal, we
could have used a probabilistic approach in which the likelihood of
observing a given color in the skin class versus that of observing the
same color in the non-skin class would have been estimated. Here, we
empirically define an acceptable hue/saturation interval for our test
image (remember that the 8-bit version of the hue goes from 0 to 180
and saturation goes from 0 to 255):
// detect skin tone
cv::Mat mask;
detectHScolor(image, 160, 10, // hue from 320 degrees to
20 degrees
25, 166, // saturation from ~0.1 to
0.65
mask);
// show masked image
cv::Mat detected(image.size(), CV_8UC3, cv::Scalar(0, 0,
0));
image.copyTo(detected, mask);The following detection image is obtained as the result:
Note that, for simplicity, we have not considered color brightness in the
detection. In practice, excluding brighter colors would have reduced the
possibility of wrongly detecting a bright reddish colors as skin.
Obviously, a reliable and accurate detection of skin color would require
a much more elaborate analysis. It is also very difficult to guarantee
good detection across different images because many factors influence
color rendering in photography, such as white balancing and lighting
conditions. Nevertheless, as shown here, using hue/saturation
information as an initial detector gives us acceptable results.
