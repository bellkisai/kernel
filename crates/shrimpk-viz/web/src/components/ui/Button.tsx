import { type ButtonHTMLAttributes, forwardRef } from "react";

const FOCUS_RING =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-canvas";

const variants = {
  primary: "bg-accent hover:bg-accent-hover active:bg-accent-active text-text-primary",
  secondary: "bg-overlay hover:bg-elevated text-text-primary",
  ghost: "bg-transparent hover:bg-overlay text-text-primary",
  danger: "bg-error/20 hover:bg-error/30 text-error",
} as const;

const sizes = {
  sm: "h-7 px-2 text-caption",
  md: "h-8 px-3 text-body",
  lg: "h-9 px-4 text-body",
} as const;

export type ButtonVariant = keyof typeof variants;
export type ButtonSize = keyof typeof sizes;

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "primary", size = "md", className = "", children, disabled, ...props }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled}
        className={[
          "inline-flex items-center justify-center gap-1.5 rounded font-medium",
          "transition-colors duration-micro",
          "motion-reduce:transition-none",
          variants[variant],
          sizes[size],
          FOCUS_RING,
          disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
          className,
        ].join(" ")}
        {...props}
      >
        {children}
      </button>
    );
  },
);

Button.displayName = "Button";
