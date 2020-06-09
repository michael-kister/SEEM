function accum = binomi(n, k)

accum = 1;

if k > n    
    accum = 0;
    return;
end

if k > n/2
    k = n - k;
end

for i = 1:k
    accum = accum * (n - k + i) / i;
end

%accum = accum + 0.5;