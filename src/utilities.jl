tuplize(x) = tuple(x)
tuplize(x::Tuple) = x

tuplemap(f, x) = f(x)
tuplemap(f, x::Tuple) = map(f, x)
