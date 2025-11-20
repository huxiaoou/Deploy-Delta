rm -r build/*
python setup.py build_ext clean
cp build/lib.linux-x86_64-3.9/math_tools.cpython-39-x86_64-linux-gnu.so solutions
rm src/math_tools.c