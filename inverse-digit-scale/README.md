# Absorb the digit division into the inverse transform

The two signs of each prime previously completed their inverse transforms before their sum and difference recovered two digits. The high digit then needed one multiplication by the reciprocal of the digit twist. The inverse transform is linear. Take the sum and difference first and carry that constant through the high digit's transform.

At butterfly distance d, both inputs have the same low d−1 bit mask. If those bits are zero, neither input has received the constant. Apply it to the upper output by precombining it with the existing transform root. If an earlier bit was set, both inputs already carry the constant. Lane zero always takes lower outputs. It receives the constant in the final stage alongside the upper lanes.

This removes 31 separate scalar fixed-root products per 32 high-digit outputs, or 471,334,912 across the ranked batch. The high-digit path issues its multiplications inside the five existing butterfly levels. It reads 72.5 KiB of fixed weights at the ranked size. Compare the full judge time with the measured 35.201212 ms source. The cooperative decoder test covers both widths and compares every resulting limb against signed integer reconstruction.
