function piSet = v2pi( vSet )
% function piSet = v2pi( vSet )
%   converts stick breaks to feature probabilities

piSet = cumprod( vSet );

return
