import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Python Basics

    This lesson covers the fundamentals of the Python language: variables, data types, collections, control flow, and functions. These are the building blocks used throughout the later lessons.

    ### Comments

    Comments start with `#` and are ignored by Python. Use them to explain what your code does.
    """)
    return


@app.cell
def _():
    # This is a comment
    print("Hello, World!")  # Print a greeting
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Multiline comments use triple quotes:
    """)
    return


@app.cell
def _():
    """
    This is a multiline comment.
    It can span multiple lines.
    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variables

    Variables act as labelled containers for data. Assign values using `=`.
    """)
    return


@app.cell
def _():
    # Assigning values to variables
    message = "Hello, Python!"  # Text (string)
    count = 10  # Whole number (integer)
    price = 19.99  # Decimal number (float)
    is_active = False  # True/False value (boolean)

    # Accessing variable values
    print(message)
    print(count)
    print(price)
    print(is_active)
    return count, is_active, message, price


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Types

    Variables can hold different types of data, such as numbers (integers or floats) and text (strings). The functionality available often depends on the data type, though some behaviours are shared across types.

    - **`str` (String):** Text, enclosed in quotes (`"` or `'`). Example: `"Urbanism"`.
    - **`int` (Integer):** Whole numbers. Example: `42`, `-7`.
    - **`float` (Float):** Numbers with decimals. Example: `3.14`, `-0.001`.
    - **`bool` (Boolean):** Logical values, `True` or `False` (capitalised).

    Check a variable's type using `type()`:
    """)
    return


@app.cell
def _(count, is_active, message, price):
    print(type(message))  # <class 'str'>
    print(type(count))  # <class 'int'>
    print(type(price))  # <class 'float'>
    print(type(is_active))  # <class 'bool'>
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When working with variables, especially in Python, be cognisant of the type assigned to the variable. Misusing types can lead to issues and errors.
    """)
    return


@app.cell
def _(count, message):
    # This will raise an error because you cannot add a string to an integer
    # TypeError: unsupported operand type(s) for +: 'int' and 'str'

    count + message
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Arithmetic

    Python supports standard arithmetic operations:
    """)
    return


@app.cell
def _():
    # Arithmetic operators: +, -, *, /, **, %
    a = 10
    b = 3

    print(a + b)  # Addition: 13
    print(a - b)  # Subtraction: 7
    print(a * b)  # Multiplication: 30
    print(a / b)  # Division (float result): 3.333...
    print(a**b)  # Exponentiation: 1000
    print(a % b)  # Modulus (remainder): 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Python follows the standard order of operations (often remembered by acronyms like BODMAS or PEMDAS). For example:
    """)
    return


@app.cell
def _():
    _result = 2 + 3 * 4 - 5 / 2
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you're ever unsure about the order, it's good practice to use parentheses `()` to make your intention explicit.
    """)
    return


@app.cell
def _():
    _result = (2 + 3) * 4 - 5 / 2
    print(_result)
    return


@app.cell
def _():
    _result = 2 + 3 * (4 - 5) / 2
    print(_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Strings

    Strings are very versatile and have a number of built-in methods for common use cases.
    """)
    return


@app.cell
def _():
    first_name = "Ada"
    last_name = "Lovelace"
    full_name = first_name + " " + last_name
    print(full_name)  # Output: Ada Lovelace
    return (full_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:**
    > A method is like a function that's associated with a particular object or type (like a string). For example, `str.upper()` is a method of the `str` class that converts a string to uppercase. You access methods using dot notation (`object.method()`) and execute them by using parentheses. If a method requires options or parameters, they go inside the parentheses.

    Common built-in methods include converting text to lower or upper case, or removing leading/trailing whitespace and characters.
    """)
    return


@app.cell
def _(full_name):
    # String methods
    print(full_name.lower())  # Lowercase: "ada lovelace"
    print(full_name.upper())  # Uppercase: "ADA LOVELACE"
    print(full_name.replace(" ", "_"))  # Replace spaces with underscores
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### F-Strings

    F-strings (formatted string literals) offer a convenient way to embed expressions inside string literals for formatting. Simply prefix the string with an `f` and write expressions in `{}`.
    """)
    return


@app.cell
def _():
    city = "Berlin"
    population = 3800000
    info = f"The city of {city} has a population of {population}."
    print(info)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercises

    - Create a variable for a number. Print its type using `type()`.
    - Create a variable for your name, then print a statement using an F-string to say "Hello, [your name]!".
    - Perform all basic arithmetic operations (+, -, \*, /, %) on two numbers and print the results. Experiment with parentheses `()` to change the order of operations.

    ## Collections

    Often, you need to work with multiple pieces of data at once. Python provides several ways to group data, each with its own characteristics.

    | Type       | Ordered | Unique       | Mutable | Example Syntax     | Example Use Case              |
    | ---------- | ------- | ------------ | ------- | ------------------ | ----------------------------- |
    | List       | Yes     | No           | Yes     | `[1, 2, 3]`        | Shopping list, coordinates    |
    | Tuple      | Yes     | No           | No      | `(1, 2, 3)`        | Fixed coordinates, RGB colour |
    | Set        | No      | Yes (Values) | Yes     | `{1, 2, 3}`        | Unique tags, deduplication    |
    | Dictionary | Yes     | Yes (Keys)   | Yes     | `{"a": 1, "b": 2}` | Lookup tables, data records   |

    ### Lists

    Use lists when the order of items is important and you might need to change them later. Lists are _mutable_ (meaning they can be changed after creation). Think of them like a row of mailboxes: you can change what's inside a mailbox. Access items using their _index_ (position, starting from 0) in square brackets `[]`. Negative indexing is also handy, where `-1` refers to the last item, `-2` to the second last, and so on.

    Create a list using square brackets []:
    """)
    return


@app.cell
def _():
    planets = ["Mercury", "Venus", "Earth", "Mars"]

    planets
    return (planets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Access items by index (position, starting from 0)
    """)
    return


@app.cell
def _(planets):
    print(planets[0])  # Mercury
    print(planets[2])  # Earth
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reverse index
    """)
    return


@app.cell
def _(planets):
    print(planets[-1])  # Mars
    print(planets[-3])  # Venus
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add an item to the end
    """)
    return


@app.cell
def _(planets):
    planets.append("Jupiter")

    planets
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:**
    > Python has several useful built-in functions. One is `len()`, which we'll use below. It returns the length (i.e., the number of items) of an object like a list or a string.

    Get the number of items
    """)
    return


@app.cell
def _(planets):
    len(planets)  # Output: 5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lists are mutable (can be changed after creation)
    """)
    return


@app.cell
def _(planets):
    planets[0] = "Fast Planet"  # Change the first item
    planets[3] = "Red Planet"  # Update Mars

    planets
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Common list operations
    """)
    return


@app.cell
def _(planets):
    planets.insert(1, "New Planet")  # Insert at a specific index
    print(planets)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Remove item at index 2 and get its value
    """)
    return


@app.cell
def _(planets):
    removed_planet = planets.pop(2)
    print(f"Removed: {removed_planet}")
    print(planets)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dictionaries

    Use dictionaries when you need to associate _values_ with unique _keys_ (like looking up a word in a dictionary to find its definition). They are very efficient for retrieving values when you know the key.

    Once data is stored, you can easily retrieve or update a value using its key.

    Dictionaries are defined using curly braces `{}`. Each item is a key-value pair, separated by a colon `:`. The key is on the left, and the value is on the right.

    Similar to lists, dictionaries use square brackets `[]` to access values, but instead of a numerical index, you use the key.
    """)
    return


@app.cell
def _():
    # Create a dictionary using curly braces {}
    # Format: {key1: value1, key2: value2}
    building_info = {"type": "Residential", "floors": 5, "year_built": 1998}
    building_info
    return (building_info,)


@app.cell
def _(building_info):
    # Access values using keys
    print(building_info["type"])  # Residential
    print(building_info["floors"])  # 5
    return


@app.cell
def _(building_info):
    # Add a new key-value pair
    building_info["has_elevator"] = True
    print(building_info)
    return


@app.cell
def _(building_info):
    # Dictionaries are mutable
    building_info["year_built"] = 2000  # Update a value
    print(building_info)
    return


@app.cell
def _(building_info):
    # Common dictionary operations
    print(building_info.keys())  # Get all keys
    print(building_info.values())  # Get all values
    print(building_info.get("floors"))  # Get value
    print(building_info.get("address"))  # Returns None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tuples

    Use tuples for ordered collections of items that shouldn't change after creation (e.g., coordinates). Tuples are _immutable_, meaning they cannot be altered once created.
    """)
    return


@app.cell
def _():
    # A tuple of coordinates
    point = (10.5, 25.3)
    print(point)
    print(point[0])  # Access items like lists
    return (point,)


@app.cell
def _(point):
    # This would cause a TypeError! Tuples are immutable.
    point[0] = 11.0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tuples are often used to return multiple values from functions or for unpacking (we will explain functions later).
    """)
    return


@app.function
def get_coordinates():
    return (10.5, 25.3)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a common pattern in Python, especially when dealing with functions that return multiple values. You can unpack the tuple into separate variables.
    """)
    return


@app.cell
def _():
    coords = get_coordinates()
    _lat, _lon = coords  # Tuple unpacking
    print(f"Latitude: {_lat}, Longitude: {_lon}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above can also be done directly:
    """)
    return


@app.cell
def _():
    _lat, _lon = get_coordinates()
    print(f"Latitude: {_lat}, Longitude: {_lon}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The below would raise an error because the tuple only has two items, but we are trying to unpack it into three variables:
    """)
    return


@app.cell
def _():
    _lat, _lon, z = get_coordinates()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sets

    Use sets when you need a collection of unique items, and the order of those items isn't important.
    """)
    return


@app.cell
def _():
    # Duplicate "London" is ignored
    unique_cities = {"London", "Paris", "Berlin", "London"}
    # Output might be in any order, e.g., {'Berlin', 'London', 'Paris'}
    print(unique_cities)
    return (unique_cities,)


@app.cell
def _(unique_cities):
    # Check for membership
    print("Paris" in unique_cities)  # True
    return


@app.cell
def _(unique_cities):
    # Add an item
    unique_cities.add("Rome")
    print(unique_cities)
    return


@app.cell
def _():
    # Set operations
    set1 = {1, 2, 3}
    set2 = {3, 4, 5}
    print(set1.union(set2))  # All items from both: {1, 2, 3, 4, 5}
    print(set1.intersection(set2))  # Items in both: {3}
    print(set1.difference(set2))  # Items in set1 but not set2: {1, 2}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Collection Exercises

    - Create a dictionary representing a building with keys for `name`, `height`, and `year_built`. Print the building's height.
    - Create a list of five city names. Use a loop to print each city name in uppercase (hint: `"text".upper()`).

    ## Control Flow

    Control flow structures are fundamental to programming. They allow your code to make decisions (using `if`/`elif`/`else`) and repeat actions (using `for`/`while` loops).

    ### Conditionals

    Conditionals allow your program to execute different blocks of code based on whether certain conditions are `True` or `False`. They use comparison operators to evaluate conditions. This conditional logic enables your program to adapt its behaviour to various situations.
    """)
    return


@app.cell
def _():
    # Comparison operators return boolean values
    print(5 == 5)  # Equal to: True
    print(5 != 3)  # Not equal to: True
    print(5 > 3)  # Greater than: True
    print(5 < 3)  # Less than: False
    print(5 >= 5)  # Greater than or equal to: True
    print(5 <= 3)  # Less than or equal to: False
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:**
    > ### Indentation
    >
    > Indentation is non-negotiable in Python: it defines code blocks. Always use consistent spacing (the common convention is 4 spaces per indentation level) and avoid mixing spaces with tabs. Code blocks are introduced by a colon (`:`) and can be nested.
    >
    > ```python
    > # FYI - we will explain loops next
    > for i in range(3):
    >     # Example of indentation
    >     if i == 2:
    >         # Example of nested indentation
    >         print(f"{i} is two")
    >     else:
    >         print(f"{i} is not two")
    > ```

    Use `if`, `elif` (short for 'else if'), and `else` to direct the flow of execution.
    """)
    return


@app.cell
def _():
    population_density = 5000  # people per sq km

    if population_density > 10000:
        print("Very high density")
    elif population_density > 3000:
        print("High density")
    elif population_density > 1000:
        print("Medium density")
    else:
        print("Low density")
    return (population_density,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Conditions can be combined using `and` and `or`.
    """)
    return


@app.cell
def _(population_density):
    # Combining conditions with 'and', 'or'
    floors = 12

    if population_density > 3000 and floors > 10:
        print("High density and tall buildings")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For identity checks (e.g., if a value `is None`), use `is` and `is not`.
    """)
    return


@app.cell
def _():
    # Checking for None (often used for missing data)
    maybe_data = None  # Represents absence of a value

    if maybe_data is None:
        print("Data is missing.")
    else:
        print(f"Data found: {maybe_data}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For checking if a value is in a collection (like a list or set), use `in` and `not in`.
    """)
    return


@app.cell
def _():
    # Check if a value is in a list
    cities = ["London", "Paris", "Berlin"]
    if "Paris" in cities:
        print("Paris is in the list of cities.")
    if "Rome" not in cities:
        print("Rome is not in the list of cities.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loops

    Loops are fantastic for avoiding repetitive code. They are essential for processing items in collections and can be combined with conditionals to apply logic selectively to each item.

    #### `for` Loops

    Use `for` loops to iterate over each item in a sequence (such as a list, tuple, or string). In each pass (iteration), the loop variable (e.g., `planet` in the example below) takes the value of the current item from the sequence.
    """)
    return


@app.cell
def _(planets):
    # Loop through a list
    for planet in planets:
        print(f"Checking planet: {planet}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `while` Loops

    Use `while` loops when you don't know exactly how many times to loop in advance, but you know the condition under which the loop should continue running.
    """)
    return


@app.cell
def _():
    # Countdown
    countdown = 3
    while countdown > 0:
        print(countdown)
        # Decrease countdown (essential to avoid infinite loop!!!)
        countdown = countdown - 1
    print("Blast off!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Caution:** Ensure the `while` loop's condition eventually becomes `False`, otherwise it will run forever (an infinite loop)!

    ### Exercise

    Create a list named `numbers` containing the integers from 1 to 10. Then, write a loop that iterates through this list. For each number, if it is even, print the number.

    **Hint:** The modulo operator (`%`) gives the remainder of a division. A number is even if `number % 2 == 0`.

    ## Functions

    Functions help you organise your code into logical, reusable blocks. This improves readability, makes debugging easier, and simplifies maintenance. If you find yourself copying and pasting code, that's often a good sign you could write a function instead.
    """)
    return


@app.function
# Define a function with parameters (inputs)
def calculate_density(population, area, unit="sq km"):
    """Calculates population density.

    It's good practice to include a *docstring* (documentation string) right after
    the def line to explain what the function does, its parameters, and what it returns.

    Args:
        population (int): The total population.
        area (float): The total area.
        unit (str, optional): The unit for the area. Defaults to "sq km".

    Returns:
        float: The calculated density, or 0 if area is non-positive.
    """
    if area <= 0:
        print("Area must be positive to calculate density.")
        return 0  # Return 0 or raise an error for invalid input
    density = population / area
    return density


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Parameters:** These are the names listed in the function definition's parentheses (e.g., `population`, `area`).
    - **Arguments:** These are the actual values you pass to the function when you call it.
    - **Default Arguments:** You can give parameters default values (e.g., `unit="sq km"`). If you don't provide an argument for that parameter when calling the function, the default value is used.
    - **Scope:** Variables defined inside a function are typically _local_ to that function, meaning they only exist and can be accessed within that function.
    - **Docstrings:** As mentioned, the optional triple-quoted string right after the `def` line. They are used to document what the function does, its arguments (often listed under an `Args:` section), and what it returns (under a `Returns:` section).
    - **Return Statement:** The `return` statement is used to send a value back from the function to the place where it was called. If a function doesn't have a `return` statement, or if the `return` statement is used without a value, it implicitly returns `None`.

    Call the function with arguments (values for the parameters)
    """)
    return


@app.cell
def _():
    density1 = calculate_density(1000000, 50)
    print(f"Density 1: {density1:.2f} people per sq km")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calling the function with explicit named arguments allows you to specify which parameter each value corresponds to, making your code clearer and more readable. This is especially useful when a function has many parameters or some have default values.
    """)
    return


@app.cell
def _():
    # Can name arguments
    density2 = calculate_density(population=500000, area=100, unit="square kilometres")
    print(f"Density 2: {density2:.2f} people per square kilometres")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While functions don't always have to return a value (e.g., a function that just prints something), they often do.

    ### Function Exercise

    Write a function called `classify_building` that takes a `height` parameter and returns `"low-rise"` if height < 10, `"mid-rise"` if 10–25, and `"high-rise"` if > 25. Test it with three different heights.

    ## Importing Modules

    Python has a vast collection of _modules_ that provide ready-to-use tools, so you often don't need to reinvent the wheel. If you encounter an unfamiliar function, it's a good idea to check the `import` statements, usually found at the top of the file, as the function might come from an imported module.

    `import module_name` makes the functions and variables within that module available, accessed via `module_name.function_name`.
    """)
    return


@app.cell
def _():
    import math  # Import the built-in math module

    radius = 5
    area = math.pi * (radius**2)  # Use math.pi
    print(f"Circle Area: {area:.2f}")
    return math, radius


@app.cell
def _(math, radius):
    circumference = 2 * math.pi * radius
    print(f"Circle Circumference: {circumference:.2f}")
    return


@app.cell
def _():
    # You can also import specific methods from a module
    from math import sqrt  # Import only the square root function

    print(f"Square root of 16 is {sqrt(16)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In later sections, we'll be making extensive use of powerful libraries such as `geopandas` and `numpy`, which are imported as modules.

    ## Common Errors

    Errors are a normal part of programming. Learning to read and understand error messages is a key skill for fixing them. If you get stuck, searching for the error message online is often helpful; chances are, someone else has encountered the same problem. You can also ask an LLM to help explain what an error message means.

    - **`NameError: name '...' is not defined`**: You tried to use a variable before assigning a value to it, or you misspelled the variable name.
    - **`TypeError: unsupported operand type(s) for ...`**: You tried to perform an operation on incompatible data types (e.g., adding a string to an integer: `"Hello" + 5`).
    - **`SyntaxError: invalid syntax`**: You made a mistake in the Python grammar (e.g., missing colon `:`, mismatched parentheses `()`).
    - **`IndexError: list index out of range`**: You tried to access an item in a list using an index that doesn't exist (e.g., accessing `my_list[5]` when the list only has 3 items).
    - **`ImportError: No module named '...'` or `ModuleNotFoundError: No module named '...'`**: You tried to import a module that Python can't find. It might be misspelled or not installed.

    **Debugging Tip:** When you encounter an error, the last line of the message usually tells you the specific type of error. The lines mentioned in the traceback help you find where the problem occurred in your code. Sprinkling `print()` statements in your code to check the values of variables at different stages can also be a very effective way to debug. And remember, it's common to fix one error only to find another; just tackle them one by one.

    ## Summary

    - Python variables store data of different types: strings, integers, floats, and booleans.
    - Collections (lists, dictionaries, tuples, sets) let you group related data, each with different characteristics for ordering, uniqueness, and mutability.
    - Control flow with `if`/`elif`/`else` and `for`/`while` loops lets your code make decisions and repeat actions.
    - Functions organise reusable logic, accept parameters, and return results.
    - Modules extend Python's capabilities; import them to access ready-made tools.
    - Errors are normal; reading the traceback and error type is the key to debugging.

    Next: [Spatial Data](https://cityseer.benchmarkurbanism.com/start/3-spatial) introduces geometric objects with Shapely.
    """)
    return


if __name__ == "__main__":
    app.run()
