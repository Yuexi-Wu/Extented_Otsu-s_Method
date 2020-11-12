**Project description:** In the original *Otus’s method* for automatic thresholding, we week to find threshold *t* that minimizes the weighted sum of within-group variances for the background and foreground pixels that result from thesholding the grayscale image at value *t*:

![\sigma _{w}^{2}(t)=\omega _{0}(t)\sigma _{0}^{2}(t)+\omega _{1}(t)\sigma _{1}^{2}(t)](https://wikimedia.org/api/rest_v1/media/math/render/svg/a54fa4d7191375eb50b1400ea63d80f3fb1146b2)

where ![{\displaystyle {\begin{aligned}\omega _{0}(t)&=\sum _{i=0}^{t-1}p(i)\\[4pt]\omega _{1}(t)&=\sum _{i=t}^{L-1}p(i)\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/64455eb76bfcd1597e6ec70d48d11aacc86f2982)

and *p(i)* is the normalized histogram (original histogram divided by total number of pixels in the image) and 𝐺 is the total number of gray level values. 

Extend the Otsu’s method to automatically determine three thresholds 𝑡𝑡1, 𝑡𝑡2 and 𝑡𝑡3 for dividing the histogram of the input image into four intervals, [0, 𝑡1], (𝑡1, 𝑡2], (𝑡2, 𝑡3] and (𝑡3 , *G-*1], so that the weighted sum of the within-group variances of the four resulting regions is minimized.

The only library functions allowed to use are those for the *reading, writing* and *displaying* of images.

**Testing your program**: Test images of size *N* X *M* in bitmap (.*bmp*) format will be provided on NYU Classes for you to test your program.