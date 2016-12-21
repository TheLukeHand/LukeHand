function bioinfochecknargin(numArgs,low,name)
%BIOINFOCHECKNARGIN Validate number of input arguments
%
%   BIOINFOCHECKNARGIN(NUM,LOW,FUNCTIONNAME) throws an MException if the
%   number of input arguments NUM to function FUNCTIONNAME is less than the
%   minimum number of expected inputs LOW.
% 
%   Example
%      bioinfochecknargin(nargin, 3, mfilename)
%
%   See also MFILENAME, NARGCHK, NARGIN, NARGOUT, NARGOUTCHK.

%   Copyright 2007  The MathWorks, Inc.
%   $Revision: 1.1.6.2 $  $Date: 2007/10/05 18:31:54 $
    
if numArgs < low
        xcptn = MException(sprintf('Bioinfo:%s:NotEnoughInputs',name),...
        'Not enough input arguments.');
    xcptn.throwAsCaller;
end
