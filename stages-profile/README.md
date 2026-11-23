# Three-stream batch profile

This diagnostic runs the selected three-stream kernels and workspace dependencies. Ten event timestamps per polynomial product delimit input conversion, the forward transform, the middle product, the inverse transform and wide reconstruction. All five stage intervals begin after their waits. The output reports how long each stage ran alone and how long two or three stages overlapped over the entire batch. Event spans include scheduling delays. Compare the instrumented batch with the selected control to measure the cost of the timestamps.
