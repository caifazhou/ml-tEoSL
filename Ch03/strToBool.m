function [y] = strToBool(x)

% x: cell array
y = zeros(size(x, 1),1);
for ii = 1:1:size(x, 1)
    if strcmp(x{ii,1},'T')
        y(ii,1) = 1;
    else
        y(ii,1) = 0;
    end;
end;