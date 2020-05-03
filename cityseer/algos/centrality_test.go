package main

import (
    /*
typedef struct foo{
int a;
int b;
int c;
int d;
int e;
int f;
} foo;
*/
    "C"
)

func main() {}

//export Foo
func Foo(t []int) C.foo {
    return C.foo{}
}