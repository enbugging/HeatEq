
# Heat Equation in CUDA

Final project for the course CSE305 - Concurrent Programming. The aim is to solve two-dimensional heat equation with explict and implicit method, and with speedup using CUDA.

Recall the heat equation

![equation](https://latex.codecogs.com/svg.image?%20%5Cfrac%7B%5Cpartial%20U%7D%7B%5Cpartial%20t%7D%20=%20C%5Cleft(%5Cfrac%7B%5Cpartial%5E2%20U%7D%7B%5Cpartial%20x%5E2%7D%20&plus;%20%5Cfrac%7B%5Cpartial%5E2%20U%7D%7B%5Cpartial%20y%5E2%7D%5Cright))

where C is the thermal conductivity constant.


## Compilation

Compilation requires CUDA Compiler `nvcc` that supports C++11 and later.
The package has `CMakeLists.txt` configured, so compilation simply amounts to 

```
cmake . -Bbuild
cmake --build build
```

after which the executable `heat` or `heat.exe` is available in folder `build`, and running it requires the command `./build/heat` 
or `./build/heat.exe`.

## Usage

The execution file has basic interface, of the form `./heat [mode] [nx] [ny] [t] [dt] [dx] [dy] [C] [initial] [number of random initials]` (or `./heat.exe` for Windows), with the following details:

- `mode`: default to be `0`, with 
    - `0` for explicit Euler, sequential mode,
    - `1` for implicit Euler, sequential mode,
    - `2` for explicit Euler, parallelised mode,
    - `3` for implicit Euler, parallelised mode,
    - `4` for explicit Euler, parallelised mode with shared memory,
    - `5` for implicit Euler, parallelised mode with shared memory;
- `nx` and `ny`: two dimensions of the grid, default to be `2000`;
- `t`: time of simulation, in seconds. Default to be `1`;
- `dx` and `dy`: step size for Δx and Δy. Default to be `0.1`. If not enough parameters are provided, then `dx = dy`;
- `dt`: step size for Δt, in seconds. Default to be `0.0005`;
- `C`: thermal conductivity constant, default to be `1`;
- `initial`: initial condition, default to be `0`, with
    - `0` for a point at the center;
    - `1` for a ring;
    - `2` for a square;
    - `3` for multiple random points;
- `number of random initials`: number of random points if `initial = 3`. Default to be 5.
## License

[MIT](https://choosealicense.com/licenses/mit/)

Copyright (c) 2023 Nguyen Doan Dai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.