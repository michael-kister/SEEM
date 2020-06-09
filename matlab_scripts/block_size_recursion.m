function [block_sizes, block] = block_size_recursion(block_sizes, block, widths, size, deg, deg_in, var_in)

for v = 1:5
    
    if deg_in < deg-1
        
        [block_sizes, block] = block_size_recursion(block_sizes, block, widths, size*widths(v), deg, deg_in+1, v);
        
    else
        
        block_sizes(block) = size * widths(v);
        
        block = block + 1;
        
    end
    
end


