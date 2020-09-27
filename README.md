# Vulkan-Rays

Progressive path tracing in Vulkan compute.

The Vulkan parts implement a trivial graphics pipeline that just displays a float image. A compute shader paints the image with one invocation per pixel.

The path tracing is loosely following [Peter Shirley's Raytracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) series (at the moment less loosely) and [Alan Wolfe's tutorial series](https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/).
So far, it implements

* Progressive refinement over time 
* Probabilistic path tracing with diffusion, reflection, and transmission
* Primitives: spheres, parametric checkered quads (because that's somewhat mandatory for raytracer test images)

<img src="https://github.com/jeweg/vulkan-rays/raw/master/screenshots/1.png">

<img src="https://github.com/jeweg/vulkan-rays/raw/master/screenshots/2.png">