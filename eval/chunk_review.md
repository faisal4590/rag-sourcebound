# Chunk review (Level 2 human read)

Issue #62. 20 chunks drawn with `random.Random(42).sample` from `data/chunks/bca146b9df2d0a2b.jsonl`
(202 chunks, `chunk_config_hash d4ebab9ed10ca7e7`), listed in id order. Answer each question with
`yes` or `no` and a short note. Target: each question scores at least 18 of 20.

Second round, 2026-10-11, after the page-break paragraph merge and the overlap fix. The first round
(207 chunks, hash e9ca297723b994f1) scored 20, 20, 20, 19. A chunk that covers the same section and pages
as a first-round chunk carries the first-round answers, marked `(carried over, verify)`: its boundaries may
have moved by a paragraph. Chunks marked `NEW` have no first-round counterpart and need answers.

Scored 2026-10-11: 20, 20, 20, 19 of 20. All four questions pass.

| Question | Yes | No |
|---|---|---|
| 1. One sentence | 20 | 0 |
| 2. Clean start and end | 20 | 0 |
| 3. Code complete | 20 | 0 |
| 4. One sub-topic | 19 | 1 |

## bca146b9df2d0a2b:02:0006 (carried over, verify)

- Chapter 02 New Versions > New Versions | printed pages 13-13 | 129 tokens | kind `callout`
- section_titles: ['New Versions']

```text
What happened to 6?
PHP 5.6 was the latest version in the five series, with the next one being 7.0. What happened to version 6? The core team started working on it, only to realise rather late that there were significant internal implementation issues. They decided to rework the engine once again, but "PHP 6" was already being written. To avoid confusion, they decided to skip version 6 and jump straight to PHP 7.

The story of PHP 6 has become folklore within the community. If you want to know the in-depth story, you can do a quick Google search.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It covered why PHP skipped version 6. The chunk has no code.

## bca146b9df2d0a2b:02:0007 (carried over, verify)

- Chapter 02 New Versions > New Versions | printed pages 14-14 | 277 tokens | kind `prose_only`
- section_titles: ['New Versions']

```text
Even though PHP 7.0 was such a milestone, we've since moved on from it as well. PHP 7.0 is already considered old: it was released in 2015 and didn't receive updates anymore five years later. Around the late 5.x releases, PHP adopted a strict release cycle: every year brings one new version. Most versions are actively supported for two years, followed by one year of additional security support. After three years, you should update, as the version you are using doesn't get security patches anymore.

Arguably, it's even better to follow along with the latest version. There will always be minor breaking changes and deprecations, but most code can easily be updated. There even are great automated tools that will take your existing codebase, detect any upgrading errors, and fix them automatically.

Automation
One of those automated tools is called Rector which has grown in popularity over the years: https://github.com/rectorphp/rector. Rector can update your codebase automatically across several PHP versions and is a great tool to know about if you ever have to deal with legacy projects.

In the next few chapters, we'll take a deep dive into the features added in PHP 7 and PHP 8.0. Before doing that, we'll start by looking at the smaller yet significant changes in this chapter.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about the evolution of PHP versions, the importance of keeping up with the latest versions, and the use of automated tools like Rector for code updates.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes. It does not have any code blocks.

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the evolution of PHP versions and the importance of keeping up with updates, along with mentioning automated tools for code maintenance.

## bca146b9df2d0a2b:02:0008 (carried over, verify)

- Chapter 02 New Versions > Trailing commas | printed pages 15-17 | 461 tokens | kind `prose_with_code`
- section_titles: ['Trailing commas', 'Formatting numeric values', 'Anonymous classes']

```text
Trailing commas

Support for trailing commas has been added incrementally up until PHP 8.0. They are now supported in arrays, function calls, parameter lists, and closure use statements.

Trailing commas are a somewhat controversial topic among developers. Some people like them; others hate them. One argument in favour of trailing commas is that they make diffs easier. For example, imagine an array with two elements:

$array = [
    $foo,
    $bar
];

Next, you'd need to add a third:

$array = [
    $foo,
    $bar,
    $baz
];

Version control systems such as GIT would list two changes instead of one because you did make two actual changes. If you'd always used trailing commas, you wouldn't need to alter the existing lines:

$array = [
    $foo,
    $bar,
    $baz,
];

Besides version control diffs, trailing commas can also be considered "easier" to reason about because you never have to worry about adding an extra comma or not. You might not prefer this writing style, and that's fine. We'll discuss the importance of a style guide in a later chapter.

Formatting numeric values

You can use an underscore to format numeric values. This underscore is completely ignored when the code is parsed but makes long numbers easier to read by humans.

This can be especially useful in tests. Take the following example: we're testing an invoice flow and we need to pass in an amount of money. It's a good idea to store money in cents to prevent rounding errors, so using an _ makes it clearer:

public function test_create_invoice()
{
    // $100 and 10 cents
    $invoice = new Invoice(100_10);

    // Assertions
}

Anonymous classes

Anonymous classes were added in PHP 7.0. They can be used to create objects on the fly; they can even extend a base class. Here's an example in a test context:

public function test_dto_type_checking_with_arrays(): void
{
    $dto = new class(
        ['arrayOfInts' => [1, 2]]
    ) extends DataTransferObject {
        /** @var int[] */
        public array $arrayOfInts;
    };

    // Assertions
}
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about the use of trailing commas in PHP and the benefits they provide, as well as the use of underscores to format numeric values for better readability.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: No, it covers two sub-topics: trailing commas and formatting numeric values.

## bca146b9df2d0a2b:03:0022 (carried over, verify)

- Chapter 03 PHP's Type System > Dealing with null — there's another way | printed pages 38-40 | 388 tokens | kind `prose_with_code`
- section_titles: ["Dealing with null — there's another way"]

```text
Dealing with null — there's another way

I started this section saying null is called a "billion-dollar mistake", but next, I showed you three ways PHP is embracing null with fancy syntax. The reality is that null is a frequent occurrence in PHP, and it's a good thing we have syntax to deal with it in a sane way. However, it's also good to look at alternatives to using null altogether. One such alternative is the null object pattern.

Instead of one Invoice class that manages internal state about whether it's paid or not; let's have two classes: PendingInvoice and PaidInvoice. The PendingInvoice implementation looks like this:

class PendingInvoice implements Invoice
{
    public function getPaymentDate(): UnknownDate
    {
        return new UnknownDate();
    }
}

PaidInvoice looks like this:

class PaidInvoice implements Invoice
{
    // …

    public function getPaymentDate(): Date
    {
        return $this->date;
    }
}

Next, there's an Invoice interface:

interface Invoice
{
    public function getPaymentDate(): Date;
}

Finally, here are the two date classes:

class Date
{
    // …
}
class UnknownDate extends Date
{
    public function format(): string
    {
        return '/';
    }
}

The null object pattern aims to replace null with actual objects, objects that behave differently because they represent the "absence" of the real object. Another benefit of using this pattern is that classes become more representative of the real world: instead of a "date or null", it's a "date or unknown date", instead of an "invoice with a state" it's a "paid invoice or pending invoice". You wouldn't need to worry about null anymore.

$invoice->getPaymentDate()->format(); // A date or '/'

You might not like this pattern, but it's important to know that the problem can be solved this way.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: The chunk is about the null object pattern in PHP, which replaces null values with actual objects to avoid null pointer exceptions.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the null object pattern as an alternative to using null in PHP.

## bca146b9df2d0a2b:03:0023 NEW

- Chapter 03 PHP's Type System > Changing types | printed pages 41-43 | 421 tokens | kind `prose_with_code`
- section_titles: ['Changing types']

```text
Changing types

Something happened in the above code samples that I wrote down as a fact, yet it's a significant addition to PHP's type system. We were able to change method signatures during inheritance. Take a look at the Invoice interface:

interface Invoice
{
    public function getPaymentDate(): Date;
}

It declares the return type of getPaymentDate as Date, yet we changed it in our PendingInvoice to UnknownDate:

class PendingInvoice implements Invoice
{
    public function getPaymentDate(): UnknownDate
    {
        /* … */
    }
}

This powerful technique is called type variance; it's supported as of PHP 7.4. It's so powerful (and a little complex) that we'll spend a whole chapter on the topic later in this book. For now, all you need to know is that return types and input parameter types are allowed to change during inheritance, but both have different rules to follow.

Another interesting detail has to do with nullable types. There's a difference between a nullable type using ? and a default = null value; you've seen them used together already in a previous example. The difference is this: if you make a type nullable in PHP, you're still expected to pass something to that function; you can't just skip the parameter:

function createOrUpdate(?Offer $offer): void
{
    // …
}

createOrUpdate();

// Uncaught ArgumentCountError:
//     Too few arguments to function createOrUpdate(),
//     0 passed and exactly 1 expected

So by adding an explicit = null default value, you're allowed to omit the value altogether:

function createOrUpdate(?Offer $offer = null): void
{
    // …
}

createOrUpdate();

Unfortunately, because of backwards compatibility, there's a little quirk with this system. If you assign a default = null value to a variable, it'll always be nullable.

Regardless of whether the type is explicitly made nullable or not, so this will be allowed:

function createOrUpdate(Offer $offer = null): void
{
    // …
}

createOrUpdate(null);
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about type variance in PHP, which allows method signatures to change during inheritance, and the nuances of nullable types and default values.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on type variance and nullable types in PHP.

## bca146b9df2d0a2b:03:0026 NEW

- Chapter 03 PHP's Type System > Disjunctive normal form types | printed pages 46-47 | 330 tokens | kind `prose_with_code`
- section_titles: ['Disjunctive normal form types']

```text
Disjunctive normal form types

Union and intersection types added lots of benefits to the language. With the introduction of disjunctive normal form types (DNF types), you can construct even more complex types. DNF types were added in PHP 8.2, allowing you to make intersection and single types unions. This allows you to add even more granular typechecking to your code.

Let's take a look at the example we had earlier. What if we had an interface like this:

interface WithUrlSegments
{
    public function getUrlSegments(): array;
}

A class Page can now implement this class:

class Page implements WithUrlSegments
{
    //
}

We can now extend the type of our URL function as follows:

function url((WithUuid&WithSlug)|WithUrlSegments $object): string
{
    //
}

The function now allows objects which implement both the WithUuid and WithSlug or objects implementing the single WithUrlSegments interface.

The most prominent use case for DNF types will probably be to make intersection types nullable:

function url((WithUuid&WithSlug)|null $object): ?string
{
    //
}

DNF types must adhere to the rules of the DNF form, which means that you can only construct types as a union of intersections and single types. This means that a type like this is not allowed:

interface A {}
interface B {}
interface C {}

function url((A|B)&C $object): string
{
    //
}

It can be rewritten into a union of intersections as follows:

function url((C&A)&(C&B)$object): string {
    //
}
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about disjunctive normal form (DNF) types in PHP 8.2, which allow for more complex type definitions by combining union and intersection types.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the concept and usage of disjunctive normal form types in PHP.

## bca146b9df2d0a2b:03:0028 (carried over, verify)

- Chapter 03 PHP's Type System > Other types | printed pages 49-51 | 312 tokens | kind `prose_with_code`
- section_titles: ['Other types', 'Enums', 'Generics']

```text
You probably won't use the newly added null, true or false types that often. They we're mainly added to make a complete type system and fix some inconsistencies in built-in methods. An interesting case to use these new types could be as follows:

class User {
    function isAdmin(): bool
}
class Admin extends User
{
    function isAdmin(): true
    {
        return true;
    }
}

class Employee extends User
{
    function isAdmin(): false
    {
        return false;
    }
}

Enums

I waited until the end of this chapter to mention them, but that doesn't make them any less important: PHP 8.1 actually supports enum types! They look like this:

enum Status: string {
    case draft = 'draft';
    case published = 'published';
    case archived = 'archived';
}

But there's of course quite a lot to tell about them, which is why we'll wait for a future chapter to look at them in depth.

Generics

There is still one major feature missing in PHP's type system today: generics. I wish I could write about them in this book, but unfortunately, they aren't supported by the language yet. One important note to make is that static analysers do support them, so while generics are not a first-class language feature, we can use them today for static analysis.

I already mentioned them at the start of this chapter, static analysers, so I think it's time to dedicated a chapter to the topic, there's lots to tell about them!
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: The chunk is about the various types available in PHP, including static, void, mixed, never, null, false, and true types.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the different types introduced in various PHP versions and their specific uses.

## bca146b9df2d0a2b:04:0035 (carried over, verify)

- Chapter 04 Static Analysis > But Hold On… | printed pages 61-62 | 611 tokens | kind `prose_only`
- section_titles: ['But Hold On…']

```text
In fact, the static analysis community is getting so much traction these days that PhpStorm — the most popular IDE for writing PHP code — added built-in support for them. This means that the result of several type checks performed by your static analyser can be shown immediately when writing code.

Tools like Psalm, PHPStan, and Phan are great, but they also lack the eloquence you'd get from built-in language support. I've gone on and on about removing doc blocks in favour of built-in types in the previous chapter, and now we're adding them again to support static analysis. Now, to be clear: these tools also work with PHP's built-in type system (without using any doc blocks), but those doc block annotations offer a lot more functionality, because PHP's syntax doesn't limit them; they are comments, after all.

On the other hand (I've said this before in the previous chapter), there's little chance that features like generics will be added anytime soon in PHP itself since they pose such a threat to runtime performance. So if there's nothing better, we'll have to settle with doc blocks anyway if we want to use static analysis to its full extent.

If only… Can you see where I'm going with this?

What if PHP supported the generic syntax, but didn't interpret it at runtime? What if you'd need to use a static analyser to guarantee correctness (when using generics), and wouldn't worry about them when running your code. That's exactly the point of static analysis.

You might be afraid of PHP not enforcing those type checks at runtime. Still, you could also argue that static analysers are way more advanced in their capabilities, exactly because they aren't run when executing code. I don't think it's such a convoluted idea at all, and in fact, other languages already use this approach. Think about TypeScript, which has grown in popularity tremendously over the years. It's compiled to JavaScript, and all its type checks are done during that compilation phase, without running the code. Now I'm not saying we need another language that compiles to PHP; I'm only saying that static analysers are very powerful tools. If you decide to embrace them, you'll notice how you can reduce the number of tests and how rarely runtime type errors occur.

Where does that leave us now? Unfortunately, not very far. I'd recommend using a static analyser in your projects, regardless of whether you want to use its advanced annotations or not. Even without those, static analysers offer a great benefit. You've got much more certainty strange edge cases aren't possible, and you need to write fewer tests, all without ever running that code once. It's a great tool to have in your toolbox, and maybe one day, we'll see PHP fully embrace its benefits.

In action
Ready to see what static analysis can do for you? I'd recommend to go to psalm.dev and play around with their interactive playground. Some great examples show the full power of static analysis.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: The chunk is about the benefits and usage of static analysis tools in PHP, particularly how they can detect errors before code execution.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the use of static analysis tools like Psalm to identify errors in PHP code.

## bca146b9df2d0a2b:08:0055 (carried over, verify)

- Chapter 08 Attributes > Rundown | printed pages 100-101 | 316 tokens | kind `callout`
- section_titles: ['Rundown']

```text
Now imagine — because the business requires it — that some LoggableEvent objects should be handled by sending a log mail, but not all. If you're relying on the method signatures to determine what events they should handle, you will end up with something like this:

class MailLogEventSubscriber
{
    public function handleOrderCreatedEvent(
        OrderCreatedEvent $event
    ): void {
        $this->actuallyHandleTheEvent($event);
    }

    public function handleInvoiceCreatedEvent(
        InvoiceCreatedEvent $event
    ): void {
        $this->actuallyHandleTheEvent($event);
    }
    public function handleInvoicePaidEvent(
        InvoicePaidEvent $event
    ): void {
        $this->actuallyHandleTheEvent($event);
    }
    private function actuallyHandleTheEvent(
        LoggableEvent $event
    ): void {
        // …
    }
}

Using attributes, however, would allow you to write something like this:

class MailLogEventSubscriber
{
    #[
        ListensTo(OrderCreatedEvent::class),
        ListensTo(InvoiceCreatedEvent::class),
        ListensTo(InvoicePaidEvent::class),
    ]
    public function handleLoggableEvent(LoggableEvent $event)
    {
        // …
    }
}

Now, if you're building an application with only a few dozen events, the simple approach is fine. If you're dealing with thousands of events, though, it becomes cumbersome. It's those cases where I prefer the explicit approach because it saves time in the end.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: The chunk is about the implementation and usage of custom attributes in PHP, specifically in the context of event listeners.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the use of attributes for event listeners and the potential redundancy of using them when method parameters could suffice.

## bca146b9df2d0a2b:08:0057 NEW

- Chapter 08 Attributes > Rundown | printed pages 103-104 | 251 tokens | kind `prose_with_code`
- section_titles: ['Rundown']

```text
Here's the boilerplate setup, to provide a little context:

class EventServiceProvider extends ServiceProvider
{
    // In real life scenarios,
    //  we'd automatically resolve and cache all subscribers
    //  instead of using a manual array.
    private array $subscribers = [
        ProductSubscriber::class,
    ];

    public function register(): void
    {
        // The event dispatcher is resolved from the container
        $eventDispatcher = $this->app->make(EventDispatcher::class);

        foreach ($this->subscribers as $subscriber) {
            // We'll resolve all listeners
            //  (methods with the ListensTo attribute)
            //  and add them to the dispatcher.
            foreach (
                $this->resolveListeners($subscriber)
                as [$event, $listener]
            ) {
                $eventDispatcher->listen($event, $listener);
            }
        }
    }
    private function resolveListeners(string $subscriberClass): array
    {
        // Return an array with [eventName => handler] items.
    }
}

You can see we're using the as [$event, $listener] syntax in the for-loop. This is called array destructuring and something we'll also cover in an upcoming chapter.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about the setup of an event service provider in PHP, demonstrating how to register event listeners using attributes and array destructuring.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the implementation of an event service provider and the use of attributes to resolve listeners.

## bca146b9df2d0a2b:08:0059 NEW

- Chapter 08 Attributes > Rundown | printed pages 107-107 | 177 tokens | kind `prose_with_code`
- section_titles: ['Rundown']

```text
Say you're parsing controller routes and you're only interested in the Route attribute. In that case you can pass the Route class as a filter:

$attributes = $reflectionClass->getAttributes(Route::class);

There's a second parameter you can pass to getAttributes() which changes how the filtering is done. By default it'll only match attributes that exactly match the given class name. However, by using ReflectionAttribute::IS_INSTANCEOF, you're able to retrieve all attributes implementing a given interface. For example, say you're parsing container definitions that consist of several potential attributes:

$attributes = $reflectionClass->getAttributes(
    ContainerAttribute::class,
    ReflectionAttribute::IS_INSTANCEOF
);

If our Autowire attribute would implement the ContainerAttribute interface, only that one would be returned and not the Route attribute. It's a nice shorthand, built into the core.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about filtering attributes in PHP using the Reflection API, specifically how to retrieve attributes that match a given class or implement a specific interface.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the use of the Reflection API to filter and retrieve attributes in PHP.

## bca146b9df2d0a2b:08:0062 NEW

- Chapter 08 Attributes > Built-In Attributes | printed pages 112-113 | 471 tokens | kind `prose_with_code`
- section_titles: ['Built-In Attributes']

```text
Built-In Attributes

Once the base RFC for attributes had been accepted, new opportunities arose to add built-in attributes to the core. One such example is the #[Deprecated] attribute, as well as a #[Jit] attribute. I'm sure we'll see more and more built-in attributes in the future, but right now, none exist.

PhpStorm, the IDE, has done something very interesting though: they have added their own custom attributes in PhpStorm. These attributes aren't real classes in your codebase, but as we've seen you only need real attribute classes when you call $reflectionAttribute->newInstance(). The built-in attributes shipped by PhpStorm can't be instantiated, since they aren't real classes; but they can be understood by the IDE to add richer static analysis options. One example is the #[ArrayShape] attribute, which can teach PhpStorm what exactly is in an array:

#[ArrayShape([
    'key1' => 'int',
    'key2' => 'string',
    'key3' => 'Foo',
    'key3' => Foo::class,
])]
function legacyFunction(…): array

By adding such an attribute, PhpStorm now knows exactly what keys are in the array returned by legacyFunction, as well as the type of each key. I would be hesitant to write code that relies on array keys and types, though (I'd use DTOs for that). However, the #[ArrayShape] is an excellent way to document a legacy codebase, where you don't always have control over its design. There are more built-in attributes shipped by PhpStorm, by the way: #[Deprecated], #[Immutable], #[Pure], #[ExpectedValues], and even more.

It's interesting to see how PhpStorm embraces the static analysis mindset we've discussed earlier: we don't need to do any runtime checks since the static analyser (PhpStorm in this case) can tell us what we're doing wrong before running the code, thanks to attributes. We need to be careful though, that static analysis tools keep some interoperability between them. If each tool decides to implement their own version of attributes, it won't make the developer experience any better.

It'll be interesting to see what kind of attributes get added, both in the core and by third-party static analysers, and how they manage to work together.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about the introduction of built-in attributes in PHP, specifically the #[Deprecated] and #[Jit] attributes, and how PhpStorm has implemented its own custom attributes for enhanced static analysis.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on built-in attributes in PHP and their usage in static analysis, particularly in the PhpStorm IDE.

## bca146b9df2d0a2b:10:0070 NEW

- Chapter 10 Short Closures > No Multi-Line | printed pages 128-128 | 188 tokens | kind `prose_only`
- section_titles: ['No Multi-Line']

```text
No Multi-Line

You might have noticed it already: short closures can only have one expression; it may be spread over multiple lines for formatting, but it must always be one expression. The reasoning is as follows: the goal of short closures is to reduce verbosity. fn is of course shorter than function in all cases. However, it was argued that if you're dealing with multi-line functions, there is less to be gained by using short closures. After all, multi-line closures are by definition already more verbose; so being able to skip two keywords (function and return) wouldn't make much of a difference.

While I can think of many one-line closures in my projects, there are also plenty of multi-line ones, and I admit to missing the short syntax in those cases. There's hope, though: it will be possible to add multi-line short closures in the future, but we'll have to wait still a little longer.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about the limitation of short closures in PHP, which can only contain a single expression, and the reasoning behind this design choice.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the restriction of short closures to a single expression and the rationale for this limitation.

## bca146b9df2d0a2b:16:0108 (carried over, verify)

- Chapter 16 MVC Frameworks > Comparisons? | printed pages 189-192 | 389 tokens | kind `prose_with_code`
- section_titles: ['Comparisons?']

```text
This might seem like another rabbit hole at first, but we can apply the same techniques just like before. It's also good to focus on the happy path first: we can assume that's the correct way to dive. So ignore that catch block, as well as whatever is happening with dispatching RequestHandled to $this->app['events']. Based on their names, it will dispatch an event after the request is handled.

That leaves us with only two lines of code:

$request->enableHttpMethodParameterOverride();

$response = $this->sendRequestThroughRouter($request);

And reading those lines, it's clear that $this->sendRequestThroughRouter($request) is where we need to go next.

/**
 * Send the given request through the middleware / router.
 *
 * @param  \Illuminate\Http\Request  $request
 * @return \Illuminate\Http\Response
 */
protected function sendRequestThroughRouter($request)
{
    $this->app->instance('request', $request);

    Facade::clearResolvedInstance('request');

    $this->bootstrap();

    return (new Pipeline($this->app))
                ->send($request)
                ->through(
                    $this->app->shouldSkipMiddleware()
                        ? []
                        : $this->middleware
                )
                ->then($this->dispatchToRouter());
}

Next we see our request sent via a pipeline through middleware, to finally be dispatched to the router. If you're familiar with MVC frameworks, route middleware probably already rings a bell. Even if you don't know about middleware we're not blocked: $this->dispatchToRouter() is probably where we need to go.

/**
 * Get the route dispatcher callback.
 *
 * @return \Closure
 */
protected function dispatchToRouter()
{
    return function ($request) {
        $this->app->instance('request', $request);

        return $this->router->dispatch($request);
    };
}
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about understanding how the Laravel framework resolves dependencies using its dependency injection container, specifically focusing on the process of making an instance of Illuminate\Contracts\Http\Kernel and handling HTTP requests.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes, it starts at a sentence and ends at the end of the code block.

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the dependency resolution process in Laravel and how it handles HTTP requests through the kernel.

## bca146b9df2d0a2b:21:0139 NEW

- Chapter 21 The JIT > Establishing a Baseline | printed pages 236-237 | 351 tokens | kind `prose_with_code`
- section_titles: ['Establishing a Baseline']

```text
Having verified it works as expected, let's move on to our first real-life comparison. We're going to compare running our code without the JIT, with the function JIT, and the tracing JIT, both are using 100MB of memory. The page we're going to benchmark shows an overview of posts, so there's some recursion happening. We're also touching several core parts of Laravel: routing, the dependency container, as well as the ORM layer.

                                      requests/second (more is better)
No JIT                                                           63.56
Function JIT                                                     66.32
Tracing JIT                                                      69.45

Here we see the results: enabling the JIT only has a slight improvement. In fact, running the benchmarks repeatedly, the results differ slightly every time: I've even seen cases where a JIT enabled run performs worse than the non JIT'ed version. Before drawing conclusions, let's bump the memory buffer limit. We'll give the JIT a little more room to breathe with 500MB of memory instead of 100MB.

                                      requests/second (more is better)
No JIT                                                           71.69
Function JIT                                                     72.82
Tracing JIT                                                      70.11

Here we have a case of the JIT performing worse. Like I said at the beginning of this chapter: I want to measure the relative impact of the JIT on real-life web projects. It's clear from these tests that sometimes there might be benefits, but it's in no way as noticeable as the fractal example we started with. I admit I'm not surprised by that. Like I wrote before: there's very little hot code to be optimised in real-life applications. We're only rarely doing fractal-like computations.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about benchmarking the performance impact of the JIT (Just-In-Time) compiler in PHP, comparing different configurations and memory limits.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the performance comparison of PHP with and without the JIT compiler under different memory conditions.

## bca146b9df2d0a2b:22:0151 NEW

- Chapter 22 Preloading > Optimised | printed pages 253-253 | 462 tokens | kind `prose_with_code`
- section_titles: ['Optimised']

```text
Optimised

Finally, we want to compare the performance gain when we're using an optimised preloading list. For testing purposes, I started the server without preloading enabled and dumped all classes that are used within that request:

get_declared_classes();

Next, I only preloaded these classes (427 in total). Together with all their dependencies, this makes for 643 classes and 1034 functions being preloaded, occupying about 11.76 MB of memory.

These are the benchmark results for this setup:

                           requests/second            time per request
No preloading              64.79                                 771ms
Naive preloading           79.69                                 627ms
Optimised preloading       86.12                                 580ms

That's around a 25% performance gain compared to not using preloading and an 8% gain compared to using the naive approach. There's a flaw with this setup though, since I generated an optimised preloading list for one specific page. In practice, you would probably need to preload more code, if you want all your pages covered.

Another approach could be to monitor which classes are loaded how many times over several hours or days on your production server and compile a preload list based on those metrics. A package that does this is called darkghosthunter/preloader. It's definitely worth checking out.

It's safe to say that preloading — even using the naive "preload everything" approach — has a positive performance impact, also on real-life projects built upon a fullblown framework. However, there's still an important side note to be made. Real-life applications will most likely not experience a 25% increase in performance. That is because they do many more things than just booting a framework. An important thing I can think about is I/O: communicating with a database server, reading and writing to the filesystem, integrating with third party services, etc. So while preloading can optimise the booting part of your code, there still might be other areas that have a much larger impact on performance. How much exactly there is to be gained will depend on your code, server, and framework you're using. I'd say go try it out, and don't forget to measure the results.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about the performance benefits of preloading in PHP, including different approaches and their effectiveness.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the performance impact of preloading in PHP applications and the methods to optimize it.

## bca146b9df2d0a2b:25:0163 NEW

- Chapter 25 Type Variance > Type Variance | printed pages 268-269 | 360 tokens | kind `prose_with_code`
- section_titles: ['Type Variance']

```text
Here we see a problem: we can't use int $id in the retrieve method, because it violates the contract specified by WithUuid. If we'd use string $uuid, there would be the same problem but the other way around.

It's those edge cases that make contravariant types useful: PHP allows us to widen argument types, in order to fulfil both promises: one made by Repository, and one made by WithUuid. Thanks to PHP 8.0's union types, we can write retrieve's implementation like so:

class OrderRepository implements Repository, WithUuid
{
    public function retrieve(int|string $id): object { /* … */ }

    // …
}

And this code works! Of course, we need to manually deal with managing both strings and integers in our retrieve method now; still, it's good to have the option available when there's no way around the problem.

So return types are covariant, argument types contravariant. What about typed properties? They are invariant, which means you're not allowed to change property types during inheritance. It is explained clearly in the typed properties RFC why that is: "The reason why property types are invariant is that they can be both read from and written to. The change from int to ?int implies that reads from the property may now also return null in addition to integers. The change from ?int to int implies that it is no longer possible to write null to the property. As such, neither contravariance nor covariance are applicable to property types."

Before PHP 7.4, you weren't allowed to widen or narrow types, even though it would be technically correct. And while it might seem like a minor change, it's actually one that makes PHP's type system much more flexible to use.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about type variance in PHP, specifically how return types are covariant, argument types are contravariant, and typed properties are invariant, allowing for more flexible type handling in inheritance.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the concept of type variance in PHP and how it affects method signatures and property types.

## bca146b9df2d0a2b:28:0173 NEW

- Chapter 28 Event Driven Development > Event Driven Development | printed pages 284-284 | 124 tokens | kind `callout`
- section_titles: ['Event Driven Development']

```text
Service? Object? Function?
I've used three terms in one paragraph to describe "a piece of code that reacts when something happens". You might even think about calling them "micro services". For now, we won't focus on the technicalities of how these services communicate with each other and assume they are, in fact, simple objects in the same codebase, running on the same server; I'll call them plain old "services" from here on out.

By the way, have you noticed how Alan Kay's vision of what "objects" are perfectly fits this model? I like it when things come together!
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about defining the concept of "services" in event-driven development, emphasizing that they are pieces of code that react to events and can be considered as simple objects within the same codebase.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on the definition and characteristics of "services" in the context of event-driven development.

## bca146b9df2d0a2b:29:0188 NEW

- Chapter 29 Static Analysers In Practice > Error Levels | printed pages 304-304 | 240 tokens | kind `prose_with_code`
- section_titles: ['Error Levels']

```text
Error Levels

First of all, both Psalm and PHPStan allow you to configure a so-called "error level". For example, when installing Psalm, you should initialize it when first running it:

./vendor/bin/psalm --init

This command will initialize the psalm.xml config file, but will also scan your codebase, and determine what the best error level is to get started with. In Psalm, the lower the lever, the more strict it is. It goes from level 1 to level 8, with 1 being the strictest.

Inspecting psalm.xml after initialization, you can see how the error level is stored in the XML. Feel free to change it if you want to try out other levels:

<psalm
  errorLevel="4"
>
    <!-- … -->
</psalm>

PHPStan turns things around: level 0 is the most lenient, while level 9 the most strict. You can use the -l flag to override the level when analyzing your codebase:

./vendor/bin/phpstan analyse -l 6 src tests

And of course, it's also saved in the config file:

parameters:
    level: 6
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about configuring error levels in static analysis tools like Psalm and PHPStan.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on how to set and understand error levels in static analysis tools for PHP.

## bca146b9df2d0a2b:29:0189 NEW

- Chapter 29 Static Analysers In Practice > Ignore Errors | printed pages 305-306 | 210 tokens | kind `prose_with_code`
- section_titles: ['Ignore Errors']

```text
Ignore Errors

Sometimes, you simply want to ignore an error. You want to tell the static analyzer that — yes — you know this is an error; but it's ok for now. There are several ways to do this.

First of all, you can use docblocks to tell Psalm or PHPStan to specifically ignore a line:

function foo() {
    /** @phpstan-ignore-next-line */
    var_dump($undefined);
}

function bar() {
    /** @psalm-suppress UndefinedVariable */
    var_dump($undefined);
}

On the other hand, you can also globally ignore issues, for example in Psalm's config file:

<psalm>
    <issueHandlers>
        <MissingPropertyType errorLevel="suppress" />
    </issueHandlers>
</psalm>

Or in PHPStan's:

parameters:
    ignoreErrors:
        - '#Call to an undefined [a-zA-Z0-9\\_]#'

You can see how Psalm uses XML entries to specifically disable issues, while PHPStan uses regex scanning on error messages.
```

**1. Can you say what the chunk is about in one sentence, without the chunks around it?**

Answer: Yes. The chunk is about methods to ignore specific errors in static analysis tools like Psalm and PHPStan, either through docblocks or configuration files.

**2. Does it start at a sentence or heading, and end at a sentence or the end of code?**

Answer: Yes

**3. Is every code block complete, or marked as part of a group?**

Answer: Yes

**4. Does it cover one sub-topic only?**

Answer: Yes. It focuses on how to suppress or ignore errors in static analysis tools for PHP.
