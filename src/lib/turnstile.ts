const VERIFY_URL = 'https://challenges.cloudflare.com/turnstile/v0/siteverify';

/**
 * secret is undefined until the Phase 5 Turnstile widget is provisioned,
 * so this is a no-op pass-through until then.
 */
export async function verifyTurnstile(
  token: string | undefined,
  secret: string | undefined,
  remoteIp?: string,
): Promise<boolean> {
  if (!secret) return true;
  if (!token) return false;

  const formData = new URLSearchParams();
  formData.append('secret', secret);
  formData.append('response', token);
  if (remoteIp) formData.append('remoteip', remoteIp);

  const resp = await fetch(VERIFY_URL, { method: 'POST', body: formData });
  const outcome = (await resp.json()) as { success: boolean };
  return outcome.success === true;
}
