I = imread('eye_image.png');
imshow(I)

Rmin = 10;
Rmax = 20;

I_smooth = smooth(single(I(:, 70)), 20)
plot(I_smooth)