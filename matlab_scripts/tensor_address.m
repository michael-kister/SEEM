function add = tensor_address(d, multi)


%fprintf('getting address of... [ %2d , %2d ]\n\n', multi(1), multi(2));

%int i, j, max, ind;
%int add = 0;
add = 0;

%int *im = (int*) malloc(d*sizeof(int));
%int *mymulti = (int*) malloc(d*sizeof(int));
im = NaN(1,d);
mymulti = NaN(1,d);

max = 0;
%ind = d-1;
ind = d;

for i = 1:d
    
    mymulti(i) = multi(i);
    
    if mymulti(i) > max
        
        max = mymulti(i);
        
    end
    
    im(i) = 0;
    
end

for i = 1:d
    
    if mymulti(i) == max
        
        im(ind) = mymulti(i);
        
        mymulti(i) = 0;
        
        max = 0;
        
        ind = ind - 1;
        
        for j = 1:d
            
            if mymulti(j) > max
                
                max = mymulti(j);
                
            end
        end
    end
end

for i = 1:d
    
    add = add + binomi(im(i)+(i-1), i);
    
end
