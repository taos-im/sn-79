# Programming Guidelines

- Strive for idiomatic, post-C++20 style
- Adhere to the Clean Architecture principles, make code modular and testable
- Prefer composition over inheritance, make use of dependency injection (potentially via `boost-ext/di`) for better mocking capability. Inheritance has its places however, so don't be a fanatic. Follow the rule that higher level classes can (but don't have to) make use of inheritance more justifiably since the costs associated with virtual calls are neglilible compared to the tidyness they provide via the Liskov Substitution Principle.
- Adherence to SOLID is a solid target
- IMPORTANT: Be mindful of levels of abstraction, it is important to maintain a hierarchy of complexity and to understand the appropriate abstraction level from context to context. Don't be afraid to write helper functions (strive for pure functions wherever possible) within components if it makes it more easy to follow this rule.
- Fast compilation is of importance, use PIMPL when the pointer traversal cost is irrelevant and place header includes into compilation units
- Keep comments to the point, and use them judiciously and sparingly
- When designing components
    - use the typical `include/taosim/my_component`, `CMakeLists.txt` and potential compilation units folder structure. If need for further decomposition of headers, you may introduce more structure within, like `include/taosim/my_component/detail` or `include/taosim/my_component/serialization`
    - the component listfile should follow the following structure:

        ```cmake
        file(GLOB sources CONFIGURE_DEPENDS "*.cpp")
        file(GLOB_RECURSE headers CONFIGURE_DEPENDS "include/*.hpp" "include/*.h")

        add_library(mylib STATIC)
        add_library(myproj::mylib ALIAS mylib)

        # Skip this for header-only.
        target_sources(mylib PRIVATE ${sources})

        target_sources(mylib
            PUBLIC  # INTERFACE for header-only.
                FILE_SET HEADERS
                BASE_DIRS include 
                FILES ${headers}
        )

        ...
        ```

- When forward-declaring, prefer to do so at the site whenever possible, for example for PIMPL: `std::unique_ptr<struct Impl> m_impl;`
- Make use of the `range-v3` facilities wherever applicable, prefer, for example, `ranges::find_if` to `std::find_if` and avoid older style iterator loops
- If you need a non-trivial `range-v3` view to loop over, prefer to construct that view just prior to the loop and place it to a variable with suffix "View" (ex. `auto itemsView = ...`)
- Avoid reinventing the wheel, make extensive use of Boost libraries when sensible for, e.g., more-exotic string operations
- Prefer `boost::regex` to `std::regex`
- Avoid repetition related to getters and setters by making use of deducing this and perfect returning
- If you need a temporary copy to be passed to a function use `auto{}` or `auto()` at the call site
- Use `//-------------------------------------------------------------------------` as separators of logical high level code structures like function implementations, classes, and namespaces. If the structures are sufficiently small and related this can be omitted. The goal is visual clarity and effective context cues. A potential comment should be placed directly under this, and, after the comment, a blank line.
- Includes should be split into three logical blocks:
    1. project
    2. third party libraries
    3. STL
- Explicit C headers (e.g. `<fcntl.h>`, `<unistd.h>`) should be wrapped in an `extern "C"` block, with no indentation inside the block:
    ```cpp
    extern "C" {
    #include <fcntl.h>
    #include <unistd.h>
    }
    ```
- When creating factories for objects to be constructed from, say, a `pugi::xml_node`, but what might also require further setup after construction (like random seed queried from another object), prefer to create a factory where the necessary objects are injected. Create the object via the prototype pattern (constructing it first according to the xml and setting the other fields afterwards via the dependencies), and return the object to the user. The factory thus encapsulates are the managerial hurdle required for creating the object.
- Use `std::swap` and `std::exchange` where applicable to simplify code
- Prefer CRTP over virtual functions where possible, liberally mark classes and structures `final` in the case of virtual functions
- Be judicious about resource management and ownership, `std::shared_ptr` should be used sparingly, higher level objects should own, e.g., `std::unique_ptr`s and these should be passed downstream as raw pointers.
- Avoid sizable parameter lists to functions. Instead prefer simple interfaces or, when necessary, passing context structs instead. For an example refer to `exchange_service/include/taosim/exchange_service/Exchange.hpp`.
- Prefer descriptor struct passing to constructors. Keep these in such a shape that aggregate initialization of members is possible. For an example refer to `checkpoint/include/taosim/checkpoint/CheckpointManager.hpp`
- Aspire for structs to support aggregate initialization in general
- Maximum length of a line is 100 with a 5% error margin, so the absolute maximum is 105 characters
- Defaulting and deleting ctors/dtors/operators should be done in the header at the site whenever possible. Default ctors should be marked `noexcept`. Mark dtors `noexcept` also when applicable.
- Getters and setters fitting on a single line with the signature should be defined in the header at the site.
- Templated member functions implementations should be defined after the class definition (if they don't fit on a single line) one by one, to keep the class API readable.
- Avoid littering compilation units' top levels with namespace aliases
- Helper functions for external library functionaliy exist in, e.g., `xml`. Make extensive use of such to keep the code straightforward and feel free to expand such suites if a repeating need arises.
- Mark variables `const` as often as possible. Make use of, e.g., IILE if some variable needs further logic run on it before it is fully defined to keep the higher level code `const`.
    ```cpp
    // Assume colls is a std::span<const std::vector<T>>.
    // Prefer this
    const auto totalSize = ranges::accumulate(colls, 0uz, std::plus{}, ranges::size);
    // To this
    size_t totalSize{};
    for (const auto& coll : colls) {
        totalSize += coll.size();
    }
    ```
- Use `auto` liberally. Often the type of a variable is irrelevant to the degree that `auto` will only improve readability.
- Variables holding
    - lambdas should always have the type `auto` and no `const` qualifiers
    - handles to more heavy resources, like in the case of `pugi::{xml_document,node}`, should be declared without `const`
    - `std::string_view` should not be marked `const` since immutability of the underlying character sequence is inherent
- Avoid explicit casting of integer types. Make use of STL facilities like `std::ssize` to obtain a signed integer version of container sizes. Also make use of `ranges::ssize`. If explicit indexing is required, use the following template, that is, prefer signed types:
  ```cpp
  for (auto i = 0z; i < std::ssize(coll); ++i) {
      // ...
  }
  ```
- Comma-separated lists of class inheritances and member variable initializations should have the commas at the start of the line, like
    ```cpp
    class Foo final
        : public Bar
        , public Baz
    {};

    class Foo
    {
    public:
        Foo(int x, int y) noexcept
            : m_x{x}
            , m_y{y}
        {}
    
    private:
        int m_x, m_y;
    };
    ```
- Looping through msgpack maps should be done according to the following pattern in the interest of conciseness:
    ```cpp
    // The context here is conversion.
    for (const auto& [k, val] : o.via.map) {
        auto key = k.as<std::string_view>();

        if (key == "foo") {
            val.convert(v.foo);
        }
        // ...
    }
    ```
- Use `std::span<T>` and `std::span<const T>` instead of, e.g., `std::vector<T>&` and `const std::vector<T>&` when designing APIs and whenever applicable (contiguous underlying). `ranges::range auto&&` is the most capable/general option, but use it only when necessary.
- `decimal_t` literals are created via the provided `operator"" _dec` for integers, and `DEC(lit)` for decimal numbers
- Avoid `std::function`. When passing, e.g., callbacks, pass them using `<concepts>` facilities. For example, `std::function<bool(int32_t)> criterion` can be replaced with `std::predicate<int32_t> auto criterion`.
- Use `<cstdint>` types instead of plain integer types to make the byte footprint obvious. `char` is the allowed exception to this rule, which is typically more descriptive than `int8_t`.
- `size_t` should be used for unsigned size types, `ptrdiff_t` for signed ones. Prefer signed sizes as per the entry regarding `std::ssize`. `std::byte` is off-limits due to poor codegen potential; use `uint8_t` (or plain `char` as, e.g., msgpack uses) instead.

