function [dv_map, block, sizes, vars, vars2] = ...
    d_map1(dv_map, block, widths, sizes, deg, deg_in, vars, vars2)

for v = 1:5
    
    % locate using 1-index; store 0-index
    vars(deg_in+1) = v-1;
    
    % locate using 1-index; value is 1-index
    sizes(deg_in+1) = widths(v);
    
    if deg_in < deg-1
                
        [dv_map, block, sizes, vars, vars2] = d_map1(dv_map, block, widths, sizes, deg, deg_in+1, vars, vars2);
        
    else
        
        m = 1;
        
        % enter 1-index as zero
        vars2(1) = 0;
        
        % access 1-index; output 1-index
        for i = 1:vars(1)
            
            % access 1-index; store 0-index contents
            vars2(1) = vars2(1) + widths(i);
            
        end
        
        % access 1-index; convert to 1-index
        for i = 1:widths(vars(1)+1)
            
            [dv_map, vars2, m] = d_map2(dv_map, block, m, widths, sizes, deg, 1, vars, vars2);
            
            % access 1-index
            vars2(1) = vars2(1) + 1;
            
        end
        
        block = block + 1;
        
    end
    
end

            
            
            