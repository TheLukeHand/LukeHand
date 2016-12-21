function [emsg,link,dlink,ilink] = stattestlink(linkArg,dataClass)
%STATTESTLINK Test link function for GLMFIT and GLMVAL

%   Copyright 1993-2005 The MathWorks, Inc.
%   $Revision: 1.5.4.2 $  $Date: 2005/06/21 19:46:28 $

emsg = '';
link = []; dlink = []; ilink = [];

% Recognize the power link.
if isfloat(linkArg) && isscalar(linkArg)
    if linkArg == 0 % equivalent to the log link
        linkArg = 'log';
    else
        linkExponent = linkArg;
        linkArg = 'power';
    end
end

% Some inverse links are defined with limits on eta so that we always get a
% [finite|positive|(0,1)] value for mu, and so we can then evaluate the link
% and its derivative with such a mu and get a finite eta.  The link functions
% themselves all have the corresponding property intrinsically, except power
% with abs(exponent) > 1.
%
% The links that map from [0,1] have order one as the natural scale, and eps is
% a reasonable lower limit on that scale (plus it's symmetric).  We don't know
% the natural scale for the other cases, so make the limits wide.
tiny = realmin(dataClass)^.25; % keep fourth powers from under/overflowing
if ischar(linkArg)
    switch linkArg
    case 'identity'
        link = @(mu) mu;
        dlink = @(mu) ones(size(mu));
        ilink = @(eta) eta;
    case 'log'
        link = @(mu) log(mu);
        dlink = @(mu) 1 ./ mu;
        % keep mu = ilink(eta) in [tiny, 1/tiny];
        lowerBnd = log(tiny); upperBnd = -lowerBnd;
        ilink = @(eta) exp(min(max(eta,lowerBnd),upperBnd));
    case 'logit'
        link = @(mu) log(mu ./ (1-mu));
        dlink = @(mu) 1 ./ (mu .* (1-mu));
        % keep mu = ilink(eta) in (approx) [eps, (1-eps)];
        lowerBnd = log(eps(dataClass)); upperBnd = -lowerBnd;
        ilink = @(eta) 1 ./ (1 + exp(-min(max(eta,lowerBnd),upperBnd)));
    case 'probit'
        link = @(mu) norminv(mu);
        dlink = @(mu) 1 ./ normpdf(norminv(mu));
        % keep mu = ilink(eta) in [eps, (1-eps)];
        lowerBnd = norminv(eps(dataClass)); upperBnd = -lowerBnd;
        ilink = @(eta) normcdf(min(max(eta,lowerBnd),upperBnd));
    case 'comploglog'
        link = @(mu) log(-log1p(-mu));
        dlink = @(mu) 1 ./ -((1-mu) .* log1p(-mu));
        % keep mu = ilink(eta) in [eps, (1-eps)];
        lowerBnd = log(-log1p(-eps(dataClass))); upperBnd = log(-log(eps(dataClass)));
        ilink = @(eta) -expm1(-exp(min(max(eta,lowerBnd),upperBnd)));
    case {'loglog', 'logloglink'}
        link = @(mu) log(-log(mu));
        dlink = @(mu)  1 ./ (mu .* log(mu));
        % keep mu = ilink(eta) in [eps, (1-eps)];
        lowerBnd = log(-log1p(-eps(dataClass))); upperBnd = log(-log(eps(dataClass)));
        ilink = @(eta) exp(-exp(min(max(eta,lowerBnd),upperBnd)));
    case 'reciprocal'
        link = @(mu) 1 ./ mu;
        dlink = @(mu) -1 ./ mu.^2;
        % keep mu = ilink(eta) in [tiny, 1/tiny];
        lowerBnd = tiny; upperBnd = 1/lowerBnd;
        ilink = @(eta) 1 ./ min(max(eta,lowerBnd),upperBnd);
    case 'power' % linkExponent==0 (equivalent to 'log') has been weeded out already
        % keep eta = link(mu) in [tiny, 1/tiny];
        lowerBnd = tiny.^min(abs(1/linkExponent),1); upperBnd = 1/lowerBnd;
        link = @(mu) min(max(mu,lowerBnd),upperBnd).^linkExponent;
        dlink = @(mu) linkExponent * mu.^(linkExponent-1);
        % keep mu = ilink(eta) in [tiny, 1/tiny];
        lowerBnd = tiny.^min(abs(linkExponent),1); upperBnd = 1/lowerBnd;
        ilink = @(eta) min(max(eta,lowerBnd),upperBnd) .^ (1/linkExponent);
    otherwise
        emsg = 'Unrecognized link function name.';
    end

elseif iscell(linkArg)
    % A cell array of three functions is okay
    if numel(linkArg) ~= 3
        emsg = 'LINK cell array must have three components';
        return
    end

    link = linkArg{1};
    if ischar(link) && ~isempty(which(link))
        name = link; link = @(mu) feval(name,mu);
    elseif ~isa(link,'function_handle') && ~isa(link,'inline')
        emsg = 'LINK function is not valid';
        return
    end

    dlink = linkArg{2};
    if ischar(dlink) && ~isempty(which(dlink))
        name = dlink; dlink = @(mu) feval(name,mu);
    elseif ~isa(dlink,'function_handle') && ~isa(dlink,'inline')
        emsg = 'LINK function derivative is not valid';
        return
    end

    ilink = linkArg{3};
    if ischar(ilink) && ~isempty(which(ilink))
        name = ilink; ilink = @(eta) feval(name,eta);
    elseif ~isa(ilink,'function_handle') && ~isa(ilink,'inline')
        emsg = 'LINK function inverse is not valid';
        return
    end

else
    emsg = 'LINK argument is not valid.';
end
