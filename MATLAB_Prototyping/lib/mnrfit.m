function [b,dev,stats] = mnrfit(x,y,varargin)
%MNRFIT Fit a nominal or ordinal multinomial regression model.
%   B = MNRFIT(X,Y) fits a nominal multinomial logistic regression model for
%   the response Y and predictor matrix X.  X is an N-by-P design matrix with
%   N observations on P predictor variables.  Y is an N-by-K matrix, where
%   Y(I,J) is the number of outcomes of the multinomial category J for the
%   predictor combinations given by X(I,:).  The sample sizes for each
%   observation (rows of X and Y) are given by the row sums SUM(Y,2).
%   Alternatively, Y can be an N element column vector of scalar integers from
%   1 to K indicating the value of the response for each observation, and all
%   sample sizes are taken to be 1.  MNRFIT automatically includes intercept
%   (constant) terms; do not enter a column of ones directly into X.
%
%   The result B is a (P+1)-by-(K-1) matrix of estimates, where each column
%   corresponds to the estimated intercept term and predictor coefficients,
%   one for each of the first (K-1) multinomial categories.  The estimates for
%   the K-th category are taken to be zero.
%
%   MNRFIT treats NaNs in X and Y as missing data, and removes the
%   corresponding observations.
%
%   B = MMNRFIT(X,Y,'PARAM1',val1,'PARAM2',val2,...) allows you to
%   specify optional parameter name/value pairs to control the model fit.
%   Parameters are:
%
%      'model' - the type of model to fit, one of the text strings 'nominal'
%         (the default), 'ordinal', or 'hierarchical'.
%
%      'interactions' - determines whether the model includes an interaction
%         between the multinomial categories and the coefficients.  Specify as
%         'off' to fit a model with a common set of coefficients for the
%         predictor variables, across all multinomial categories.  This is
%         often described as "parallel regression".  Specify as 'on' to fit a
%         model with different coefficients across categories.  In all cases,
%         the model has different intercepts across categories.  Thus, B is a
%         vector containing K-1+P coefficient estimates when 'interaction' is
%         'off', and a (P+1)-by-(K-1) matrix when it is 'on'. The default is
%         'off' for ordinal models, and 'on' for nominal and hierarchical
%         models.
%
%      'link' - the link function to use for ordinal and hierarchical models.
%         The link function defines the relationship g(mu_ij) = x_i*b_j
%         between the mean response for the i-th observation in the j-th
%         category, mu_ij, and the linear combination of predictors x_i*b_j.
%         Specify the link parameter value as one of the text strings 'logit'
%         (the default), 'probit', 'comploglog', or 'loglog'.  You may not
%         specify the 'link' parameter for nominal models; these always use a
%         multivariate logistic link.
%
%      'estdisp' - specify as 'on' to estimate a dispersion parameter for
%         the multinomial distribution in computing standard errors, or 'off'
%         (the default) to use the theoretical dispersion value of 1.
%
%   [B,DEV] = MNRFIT(...) returns the deviance of the fit.
%
%   [B,DEV,STATS] = MNRFIT(...) returns a structure that contains the
%   following fields:
%       'dfe'       degrees of freedom for error
%       's'         theoretical or estimated dispersion parameter
%       'sfit'      estimated dispersion parameter
%       'se'        standard errors of coefficient estimates B
%       'coeffcorr' correlation matrix for B
%       'covb'      estimated covariance matrix for B
%       't'         t statistics for B
%       'p'         p-values for B
%       'resid'     residuals
%       'residp'    Pearson residuals
%       'residd'    deviance residuals
%
%   See also MNRVAL, GLMFIT, GLMVAL, REGRESS, REGSTATS.

%   References:
%      [1] McCullagh, P., and J.A. Nelder (1990) Generalized Linear
%          Models, 2nd edition, Chapman&Hall/CRC Press.

%   Copyright 2006 The MathWorks, Inc.
%   $Revision: 1.1.6.4 $  $Date: 2006/11/11 22:55:27 $

if nargin < 2
    error('stats:mnrfit:TooFewInputs', ...
          'Requires at least two input arguments.');
end

pnames = {  'model' 'interactions' 'link' 'estdisp'};
dflts =  {'nominal'             []     []     'off'};
[eid,errmsg,model,interactions,link,estdisp] = ...
                               statgetargs(pnames, dflts, varargin{:});
if ~isempty(eid)
    error(sprintf('stats:mnrfit:%s',eid),errmsg);
end

if ischar(model)
    modelNames = {'nominal','ordinal','hierarchical'};
    i = strmatch(lower(model), modelNames);
    if isempty(i)
    error('stats:mnrfit:BadModel', ...
          'The value of the ''model'' parameter must be ''nominal'', ''ordinal'', or ''hierarchical''.');
    end
    model = modelNames{i};
else
    error('stats:mnrfit:BadModel', ...
          'The value of the ''model'' parameter must be ''nominal'', ''ordinal'', or ''hierarchical''.');
end

if isempty(interactions)
    % Default is 'off' for ordinal models, 'on' for nominal or hierarchical
    parallel = strcmp(model,'ordinal');
elseif isequal(interactions,'on')
    parallel = false;
elseif isequal(interactions,'off')
    parallel = true;
elseif islogical(interactions)
    parallel = ~interactions;
else % ~islogical(interactions)
    error('stats:mnrfit:BadInteractions', ...
          'The value of the ''interactions'' parameter must be ''on'' or ''off''.');
end
if parallel && strcmp(model,'nominal')
    % A nominal model with no interactions is the same as having no predictors.
    warning('stats:mnrfit:NominalNoInteractions', ...
            'A nominal model with no category interactions is equivalent\nto a model with no predictor variables.');
    x = zeros(size(x,1),0,class(x));
end

dataClass = superiorfloat(x,y);

if isempty(link)
    link = 'logit';
elseif ~isempty(link) && strcmp(model,'nominal')
    error('stats:mnrfit:LinkNotAllowed', ...
          'You may not specify the ''link'' parameter for a nominal model.');
end
if ischar(link) && ismember(link, {'logit' 'probit' 'comploglog' 'loglog'})
    [emsg,flink,dlink,ilink] = stattestlink(link,dataClass);
else
    error('stats:mnrfit:BadLink', ...
          'The value of the ''link'' parameter must be ''logit'', ''probit'', ''comploglog'', or ''loglog''.');
end

if isequal(estdisp,'on')
    estdisp = true;
elseif isequal(estdisp,'off')
    estdisp = false;
elseif ~islogical(estdisp)
    error('stats:mnrfit:BadEstDisp', ...
          'The value of the ''estdisp'' parameter must be ''on'' or ''off''.');
end

% Remove missing values from the data.  Also turns row vectors into columns.
[anybad,wasnan,y,x] = statremovenan(y,x);
if anybad
    error('stats:mnrfit:InputSizeMismatch', ...
          'X and Y must have the same number of rows.');
end
p = size(x,2);
[n,k] = size(y);
if n == 0
    error('stats:mnrfit:NoData', ...
          'X and Y must contain at least one valid observation.');
end

if k == 1
    if min(y) < 1 || any(y ~= floor(y))
        error('stats:mnrfit:BadY', ...
             'If Y is a column vector, it must contain positive integer category numbers.');
    end
    y = accumarray({(1:n)' y},ones(dataClass));
    k = size(y,2);
    m = ones(n,1,dataClass);
else
    m = sum(y,2);
end
if parallel
    pstar = k - 1 + p;
    dfe = n * (k-1) - pstar;
else
    pstar = p + 1;
    dfe = (n-pstar) * (k-1);
end

if strcmp(model,'hierarchical')
    if nargout < 3
        [b,dev] = hierarchicalFit(x,y,m,link,n,k,p,pstar,parallel,estdisp);
    else
        [b,dev,stats] = ...
            hierarchicalFit(x,y,m,link,n,k,p,pstar,parallel,estdisp);
    end
else
    % Set up initial estimates from the data themselves
    pi = y ./ repmat(m,1,k); % the raw percentages
    pi = pi + (1/k - pi) ./ repmat(m,1,k); % shrink towards equal probabilities
    if strcmp(model,'nominal')
        [b,hess,pi] = nominalFit(x,y,m,pi,n,k,p,pstar,parallel);
    else % 'ordinal'
        z = cumsum(y(:,1:(k-1)),2);
        [b,hess,pi,gam] = ...
            ordinalFit(x,z,m,pi,flink,ilink,dlink,n,k,p,pstar,parallel);
    end

    % Deviance residuals - one for each vector observation of cell counts
    mu = pi .* repmat(m,1,k);
    D = zeros(size(y),dataClass);
    t = (y > 0); % avoid 0*log(0), but let (pi==0) & (y>0) happen
    D(t) = 2 * y(t) .* log(y(t) ./ mu(t));
    rd = sum(D,2);
    dev = sum(rd);

    if nargout > 2
        % The Pearson residuals in terms of y and pi are not equivalent to
        % those computed using z and gamma.  Use the appropriate version to
        % estimate dispersion.
        if strcmp(model,'nominal')
            r = y - pi .* repmat(m,1,k);
            rp = r ./ sqrt(pi .* (1 - pi) .* repmat(m,1,k));
            sigsq = ((k-1)/k) * sum(sum(rp .* rp)) ./ dfe; % bias corrected
        elseif strcmp(model,'ordinal')
            r = z - gam .* repmat(m,1,k-1);
            rp = r ./ sqrt(gam .* (1 - gam) .* repmat(m,1,k-1));
            sigsq = sum(sum(rp .* rp)) ./ dfe;
        end
        stats.beta = b;
        stats.dfe = dfe;
        if dfe > 0
            stats.sfit = sqrt(sigsq);
        else
            stats.sfit = NaN;
        end
        if estdisp
            stats.s = stats.sfit;
            rp = rp ./ stats.sfit;
        else
            stats.s = ones(dataClass);
        end
        stats.estdisp = estdisp;

        if ~isnan(stats.s) % dfe > 0 or estdisp == 'off'
            % bcov = inv(hess); bcov = (bcov + bcov')/2;
            bcov = linsolve(hess,eye(size(hess)),struct('SYM',true,'POSDEF',true));
            if estdisp
                bcov = bcov * sigsq;
            end
            se = sqrt(diag(bcov));
            stats.covb = bcov;
            stats.coeffcorr = bcov ./ (se*se');
            if ~parallel
                se = reshape(se,pstar,k-1);
            end
            stats.se = se;
            stats.t = b ./ se;
            if estdisp
                stats.p = 2 * tcdf(-abs(stats.t), dfe);
            else
                stats.p = 2 * normcdf(-abs(stats.t));
            end
        else
            stats.se = NaN(size(b),dataClass);
            stats.coeffcorr = NaN(numel(b),dataClass);
            stats.t = NaN(size(b),dataClass);
            stats.p = NaN(size(b),dataClass);
        end
        stats.resid = r;
        stats.residp = rp;
        stats.residd = rd;
    end
end
if nargout > 2 && any(wasnan)
    stats.resid  = statinsertnan(wasnan, stats.resid);
    stats.residp = statinsertnan(wasnan, stats.residp);
    stats.residd = statinsertnan(wasnan, stats.residd);
end


%------------------------------------------------------------------------
function [b,XWX,pi,gam] = ordinalFit(x,z,m,pi,flink,ilink,dlink,n,k,p,pstar,parallel)

kron1 = repmat(1:k-1,pstar,1);
kron2 = repmat((1:pstar)',1,k-1);

gam = cumsum(pi(:,1:(k-1)),2);
eta = flink(gam);

% Main IRLS loop
iter = 0;
iterLim = 100;
tolpos = eps(class(pi))^(3/4);
seps = sqrt(eps); % don't depend on class
convcrit = 1e-6;
b = 0;
while iter <= iterLim
    iter = iter + 1;

    % d.gamma(i,)/d.eta(i,) is actually (k-1) by (k-1) but diagonal,
    % so can store d.mu/d.eta as n by (k-1) even though it is really
    % n by (k-1) by (k-1)
    mu = repmat(m,1,k-1) .* gam;
    deta = dlink(gam) ./ repmat(m,1,k-1); % d(eta)/d(mu)
    dmu = 1 ./ deta;  % d(mu)/d(eta)

    % Adjusted dependent variate
    Z = eta + deta.*(z - mu);

    % Tridiagonal symmetric weight matrix (scaled by m)
    diagW = dmu .* dmu .* (1./pi(:,1:(k-1)) + 1./pi(:,2:k));
    offdiagW = -(dmu(:,1:(k-2)) .* dmu(:,2:k-1)) ./ pi(:,2:(k-1));

    % Update the coefficient estimates.
    b_old = b;
    XWX = 0;
    XWZ = 0;
    for i = 1:n
        W = (1./m(i)) .* (diag(diagW(i,:)) + ...
                          diag(offdiagW(i,:),1) + diag(offdiagW(i,:),-1));
        if p > 0
            % The first step for a nonparallel model can be wild, so fit
            % a parallel model for the first iteration, regardless
            if parallel || (iter==1)
                % Do these computations, but more efficiently
                % Xstar = [eye(k-1) repmat(x(i,:),k-1,1)];
                % XWX = XWX + Xstar'*W*Xstar;
                % XWZ = XWZ + Xstar'*W*Z(i,:)';
                xi = x(i,:);
                OneW = sum(W,1);
                xOneW = xi'*OneW;
                XWX = XWX + [W      xOneW'; ...
                             xOneW  sum(OneW)*(xi'*xi)];
                XWZ = XWZ + [W; xOneW] * Z(i,:)';
            else
                xstar = [1 x(i,:)];
                % Do these computations, but more efficiently
                % XWX = XWX + kron(W, xstar'*xstar);
                % XWZ = XWZ + kron(W*Z(i,:)', xstar');
                XWX = XWX + W(kron1,kron1) .* (xstar(1,kron2)'*xstar(1,kron2));
                WZ = Z(i,:)*W;
                XWZ = XWZ + WZ(1,kron1)' .* xstar(1,kron2)';
            end
        else
            XWX = XWX + W;
            XWZ = XWZ + W * Z(i,:)';
        end
    end
    b = XWX \ XWZ;

    % Update the linear predictors.
    eta_old = eta;
    if parallel
        if p > 0
            eta = repmat(b(1:(k-1))',n,1) + repmat(x*b(k:pstar),1,k-1);
        else
            eta = repmat(b',n,1);
        end
    else
        if iter == 1
            % the first iteration was a parallel fit, transform those
            % estimates to the equivalent non-parallel format.
            b = [b(1:k-1)'; repmat(b(k:end),1,k-1)];
        else
            % Convert from vector to the matrix format.
            b = reshape(b,pstar,k-1);
        end
        if p > 0
            eta = repmat(b(1,:),n,1) + x*b(2:pstar,:);
        else
            eta = repmat(b,n,1);
        end
    end

    % Update the predicted cumulative and category probabilities.
    for backstep = 0:10
        gam = ilink(eta);
        diffgam = diff(gam,[],2);
        pi = [gam(:,1) diffgam 1-gam(:,k-1)];

        % If all observations have positive category probabilities,
        % we can take the step as is.
        if all(pi(:) > tolpos)
            break;

        % Otherwise try a shorter step in the same direction.  eta_old is
        % feasible, even on the first iteration.
        elseif backstep < 10
            eta = eta_old + (eta - eta_old)/5;

        % If the step direction just isn't working out, force the
        % category probabilities to be positive, and make the cumulative
        % probabilities and linear predictors compatible with that.
        else
            pi = max(pi,tolpos);
            pi = pi ./ repmat(sum(pi,2),1,k);
            gam = cumsum(pi(:,1:k-1),2);
            eta = flink(gam);
            break;
        end
    end

    % Check stopping conditions.
    cvgTest = abs(b-b_old) > convcrit * max(seps, abs(b_old));
    if (~any(cvgTest(:))), break; end
end
if iter > iterLim
    warning('stats:mnrfit:IterOrEvalLimit', ...
            ['Maximum likelihood estimation did not converge.  Iteration limit\n' ...
             'exceeded.  You may need to merge categories to increase observed counts.']);
end


%------------------------------------------------------------------------
function [b,XWX,pi] = nominalFit(x,y,m,pi,n,k,p,pstar,parallel)

kron1 = repmat(1:k-1,pstar,1);
kron2 = repmat((1:pstar)',1,k-1);

eta = log(pi);

% Main IRLS loop
iter = 0;
iterLim = 100;
tolpos = eps(class(pi))^(3/4);
seps = sqrt(eps); % don't depend on class
convcrit = 1e-6;
b = 0;
while iter <= iterLim
    iter = iter + 1;

    mu = repmat(m,1,k) .* pi;

    % Updated the coefficient estimates.
    b_old = b;
    XWX = 0;
    XWZ = 0;
    for i = 1:n
        W = diag(mu(i,:)) - mu(i,:)'*pi(i,:);

        % Adjusted dependent variate
        Z = eta(i,:)*W + (y(i,:) - mu(i,:));

        if p > 0 % parallel models with p>0 have been weeded out
            xstar = [1 x(i,:)];
            % Do these computations, but more efficiently
            % XWX = XWX + kron(W(1:k-1,1:k-1), xstar'*xstar);
            % XWZ = XWZ + kron(Z(1:k-1)', xstar');
            XWX = XWX + W(kron1,kron1) .* (xstar(1,kron2)'*xstar(1,kron2));
            XWZ = XWZ + Z(1,kron1)' .* xstar(1,kron2)';
        else
            XWX = XWX + W(1:k-1,1:k-1);
            XWZ = XWZ + Z(1:k-1)';
        end
    end
    b = XWX \ XWZ;

    % Update the linear predictors.
    eta_old = eta;
    if parallel % parallel models with p>0 have been simplified already
        eta = repmat(b',n,1);
    else
        b = reshape(b,pstar,k-1);
        if p > 0
            eta = repmat(b(1,:),n,1) + x*b(2:pstar,:);
        else
            eta = repmat(b,n,1);
        end
    end
    eta = [eta zeros(n,1,class(eta))];

    % Update the predicted category probabilities.
    for backstep = 0:10
        pi = exp(eta);
        pi = pi ./ repmat(sum(pi,2),1,k);

        % If all observations have positive category probabilities,
        % we can take the step as is.
        if all(pi(:) > tolpos)
            break;

        % Otherwise try a shorter step in the same direction.  eta_old is
        % feasible, even on the first iteration.
        elseif backstep < 10
            eta = eta_old + (eta - eta_old)/5;

        % If the step direction just isn't working out, force the
        % category probabilities to be positive, and make the linear
        % predictors compatible with that.
        else
            pi = max(pi,tolpos);
            pi = pi ./ repmat(sum(pi,2),1,k);
            eta = log(pi);
            break;
        end
    end

    % Check stopping conditions
    cvgTest = abs(b-b_old) > convcrit * max(seps, abs(b_old));
    if (~any(cvgTest(:))), break; end
end
if iter > iterLim
    warning('stats:mnrfit:IterOrEvalLimit', ...
            ['Maximum likelihood estimation did not converge.  Iteration limit\n' ...
             'exceeded.  You may need to merge categories to increase observed counts.']);
end


%------------------------------------------------------------------------
function [b,dev,stats] = hierarchicalFit(x,y,m,link,n,k,p,pstar,parallel,estdisp)

dataClass = superiorfloat(x,y);

% Compute the sample sizes for the conditional binomial observations.  Some
% might be zero, rely on glmfit to ignore those, tell us the right dfe, and
% return NaN residuals there.
m = [m repmat(m,1,k-2)-cumsum(y(:,1:(k-2)),2)];

warnStateSaved = warning('off','stats:glmfit:IterationLimit');
[wmsgSaved,widSaved] = lastwarn;
lastwarn(''); % clear this so we can look for a new iter limit warning
needToWarn = false;
try
    if parallel
        % Same slopes for the categories, fit a single binomial model by
        % transforming the multinomial observations into conditional binomial
        % observations.
        ii = repmat(1:n,1,k-1);
        jj = repmat(1:k-1,n,1);
        dummyvars = eye(k-1,k-1,dataClass);
        xstar = [dummyvars(jj,:) x(ii,:)];
        ystar = y(:,1:k-1);
        if estdisp, estdisp = 'on'; else estdisp = 'off'; end
        if nargout < 3
            [b,dev] = glmfit(xstar,[ystar(:) m(:)],'binomial',...
                'link',link,'constant','off','estdisp',estdisp);
            needToWarn = checkForIterWarn(needToWarn);
        else
            [b,dev,stats] = glmfit(xstar,[ystar(:) m(:)],'binomial', ...
                'link',link,'constant','off','estdisp',estdisp);
            needToWarn = checkForIterWarn(needToWarn);
            stats.resid = reshape(stats.resid,n,k-1);
            stats.residp = reshape(stats.residp,n,k-1);
            stats.residd = sum(reshape(stats.residd,n,k-1),2);
            stats = rmfield(stats,'resida');
        end

    else % ~parallel
        % Separate slopes for the categories, fit a sequence of conditional
        % binomial models
        b = zeros(pstar,k-1,dataClass);
        dev = zeros(dataClass);
        if nargout < 3
            for j = 1:k-1
                [b(:,j),d] = glmfit(x,[y(:,j) m(:,j)], 'binomial','link',link);
                needToWarn = checkForIterWarn(needToWarn);
                dev = dev + d;
            end
        else
            stats = struct('beta',zeros(pstar,k-1,dataClass), ...
                           'dfe',zeros(dataClass), ...
                           'sfit',NaN(dataClass), ...
                           's',ones(dataClass), ...
                           'estdisp',estdisp, ...
                           'se',zeros(pstar,k-1,dataClass), ...
                           'coeffcorr',zeros(pstar*(k-1),dataClass), ...
                           't',zeros(pstar,k-1,dataClass), ...
                           'p',zeros(pstar,k-1,dataClass), ...
                           'resid',zeros(n,k-1,dataClass), ...
                           'residp',zeros(n,k-1,dataClass), ...
                           'residd',zeros(n,1,dataClass));
            for j = 1:k-1
                [b(:,j),d,s] = glmfit(x,[y(:,j) m(:,j)], 'binomial','link',link);
                needToWarn = checkForIterWarn(needToWarn);
                dev = dev + d;
                stats.beta(:,j) = b(:,j);
                stats.dfe = stats.dfe + s.dfe; % not n-pstar if some m's are zero
                stats.se(:,j) = s.se;
                jj = (j-1)*pstar + (1:pstar);
                stats.coeffcorr(jj,jj) = s.coeffcorr;
                stats.p(:,j) = s.p;
                stats.t(:,j) = s.t;
                stats.resid(:,j)  = s.resid;
                stats.residp(:,j) = s.residp;
                stats.residd = stats.residd + s.residd;
            end
            if stats.dfe > 0
                % Weed out the NaN residuals caused by zero conditional sizes
                % when computing dispersion.
                t = ~isnan(stats.residp(:));
                sigsq = sum(stats.residp(t) .* stats.residp(t)) ./ stats.dfe;
                stats.sfit = sqrt(sigsq);
            else
                % stats.sfit already NaN
            end
            if estdisp
                sigma = stats.sfit;
                stats.s = sigma;
                stats.residp = stats.residp ./ sigma;
                stats.se = stats.se .* sigma;
                stats.t = stats.t ./ sigma;
                stats.p = 2 * tcdf(-abs(stats.t), stats.dfe);
            else
                % stats.s already 1
            end
        end
    end
catch
    warning(warnStateSaved);
    rethrow(lasterror);
end
[wmsg,wid] = lastwarn;
if needToWarn
    warning('stats:mnrfit:IterOrEvalLimit', ...
            ['Maximum likelihood estimation did not converge.  Iteration limit\n' ...
             'exceeded.  You may need to merge categories to increase observed counts.']);
elseif ~isempty(widSaved) && isempty(wid)
    % Restore any pre-existing warning if there was not a new one.
    lastwarn(wmsgSaved,widSaved);
end
warning(warnStateSaved);

function needToWarn = checkForIterWarn(needToWarn)
[wmsg,wid] = lastwarn;
needToWarn = needToWarn || strcmp(wid,'stats:glmfit:IterationLimit');
