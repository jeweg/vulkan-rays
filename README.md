# Vulkan-Rays

Progressive path tracing in Vulkan compute.

The Vulkan parts implement a trivial graphics pipeline that just displays a float image. A compute shader paints the image by path tracing a virtual scene.

The path tracing implementation loosely follows [Peter Shirley's Raytracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) series (at the moment less loosely) and [Alan Wolfe's tutorial series](https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/).
So far, it has

* Probabilistic path tracing with diffusion, reflection, and transmission:
  * [R2 low discrepancy sequence](http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/) for sub-pixel variation of the primary rays (i.e., anti-aliasing)
  * Randomly chosen diffuse, specular, transmissive rays
  * Russian roulette and absorption-threshold-based ray termination
* Primitives: spheres, parametric checkered quads (because that seems mandatory for raytracer test images)
* Rendering into sRGB target for proper gamma correction
* Reinhard tone mapping and ACES filmic tone mapping
* Simple upright orbit camera

Libraries used: glm, GLFW, Dear ImGui, Vulkan Memory Allocator

<img src="https://github.com/jeweg/vulkan-rays/raw/master/screenshots/1.png">

<img src="https://github.com/jeweg/vulkan-rays/raw/master/screenshots/2.png">

<img src="https://github.com/jeweg/vulkan-rays/raw/master/screenshots/3.png">