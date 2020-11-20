package hello

import "fmt"

func Hello(name string) string {
    return fmt.Sprintf("hello %q from Go", name)
}