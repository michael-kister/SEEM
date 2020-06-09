function [dv_map, vars2, m] = d_map2(dv_map, block, m, widths, sizes, deg, deg_in, vars, vars2)

% access 1-index; store 0-index
vars2(deg_in+1) = 0;

% access 1-index; go up through 0-index
for i = 1:(vars(deg_in+1))
    
    % access 1-index; 
    vars2(deg_in+1) = vars2(deg_in+1) + widths(i);
    
end

% access 1-index
for v2 = 1:widths(vars(deg_in+1)+1)
    
    if deg_in < deg-1
        
        [dv_map, vars2, m] = d_map2(derivative_map, block, m, widths, sizes, deg, deg_in, vars, vars2);

    else
        
        vars22 = vars2(1:deg);
        
        %fprintf('curious about........ [ %2d , %2d ]\n', vars22(1), vars22(2));
    
        vars22 = sort(vars22);
        
        dv_map(block, m) = dv_map(block, m) * tensor_address(deg, vars22);
        
        m = m + 1;
        
    end
    
    vars2(deg_in+1) = vars2(deg_in+1) + 1;
    
end

        
