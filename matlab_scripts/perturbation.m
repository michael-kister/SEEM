
widths = [1 2 2 1 1];

block = 1;

nblock = 25;

block_sizes = zeros(1, nblock);

deg = 2;

for v = 1:5
    
    [block_sizes, block] = block_size_recursion(block_sizes, block, widths, widths(v), deg, 1, v);
    
end

n_all = sum(widths) - 1;

derivatives = NaN(nblock, n_all * max(block_sizes));

dv_map = NaN(nblock, max(block_sizes));

for i = 1:nblock
    
    derivatives(i,1:(n_all*block_sizes(i))) = 0;
    
    dv_map(i,1:block_sizes(i)) = 1;
    
end

block = 1;

vars = NaN(1, deg);

vars2 = vars;

sizes = vars;

for v = 1:5
    
    % storing an 0-index
    vars(1) = v-1;
    
    % refer to a 1-index
    sizes(1) = widths(v);
    
    [dv_map, block, sizes, vars, vars2] = d_map1(dv_map, block, widths, sizes, deg, 1, vars, vars2);
    
end
    
% eventually, we loop over each block, and in each block... the point is
% that ADOL-C organizes its results by the variable that is being
% differentiated, but I want to group results by the order of
% differentiation.








    

