# Renewable private Drive access for Phase 4E

The connected Drive app can restore private backups, but the long-running source
stream and preservation scripts use direct Drive HTTP ranges. The cloud runtime
currently supplies only `GOOGLE_DRIVE_ACCESS_TOKEN`, which expires during research.

The scripts now accept a Google OAuth refresh grant. When all three environment
bindings below are present, they mint an access token before use and again one
minute before expiry. No 401 is retried. The existing access-token path remains
available when none of the three bindings is present. A partial configuration
fails closed. Secrets are never written to receipts, logs or Git.

One-time account setup needed in the managed cloud environment:

1. Authorize an OAuth client for the Google account that owns the private source
   and backups, with offline access and a Drive scope permitting read and private
   backup upload. Store its refresh token, client ID and client secret as secure
   bindings named `GOOGLE_DRIVE_REFRESH_TOKEN`, `GOOGLE_DRIVE_CLIENT_ID`, and
   `GOOGLE_DRIVE_CLIENT_SECRET`. Do not enter their values in chat or repository.
   A Google OAuth app left in Testing mode can issue refresh tokens that expire
   after seven days; use a production consent configuration for lasting access.
2. Allow outbound HTTPS to `oauth2.googleapis.com` in the managed environment.
   Keep `www.googleapis.com` allowed for Drive API requests. Republish the
   environment and verify the three bindings are ready.
3. Run a fresh, immutable source probe receipt. Require metadata identity/version
   match and exact HTTP 206 range before resuming a date. Stop at the first auth
   failure and preserve the incomplete namespace.

This setup requires the account owner's authorization and cloud environment
configuration. The current connected Drive app's renewable credential cannot be
borrowed by the repository process. A service account would require separate
sharing of the source files and private backup destination, so it is not the
smallest setup for this user-owned Drive data.
